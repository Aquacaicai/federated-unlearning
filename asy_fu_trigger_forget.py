import copy
import itertools
import math
import queue
import random
import threading
import time
from collections import deque
import sys
import logging
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib
import matplotlib.pyplot as plt

import art
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_mnist, preprocess, to_categorical

from utils.model import FLNet
from utils.local_train import LocalTraining as BaseLocalTraining
from utils.utils import Utils
from utils.fusion import FusionAvg, FusionRetrain


class LocalTraining(BaseLocalTraining):
    """Local training with optional anchor regularization for post-unlearning."""

    def __init__(self, num_updates_in_epoch=None, num_local_epochs=1):
        super().__init__(num_updates_in_epoch=num_updates_in_epoch,
                         num_local_epochs=num_local_epochs)

    def train(self, model, trainloader, criterion=None, opt=None, lr=1e-2,
              anchor_state_dict=None, use_anchor_reg=False, anchor_beta=0.0):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if opt is None:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        if self.num_updates is not None:
            self.num_local_epochs = 1

        model.train()
        running_loss = 0.0
        device = next(model.parameters()).device
        anchor_state = None
        if use_anchor_reg and anchor_state_dict is not None:
            anchor_state = {k: v.clone().detach().to(device)
                            for k, v in anchor_state_dict.items()}

        for epoch in range(self.num_local_epochs):
            for batch_id, (data, target) in enumerate(trainloader):
                x_batch = data.to(device)
                y_batch = target.to(device)

                opt.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                if anchor_state is not None:
                    reg_loss = 0.0
                    model_state = model.state_dict()
                    for name, param in model_state.items():
                        reg_loss += torch.sum((param - anchor_state[name]) ** 2)
                    loss = loss + (anchor_beta / 2.0) * reg_loss

                loss.backward()
                opt.step()

                running_loss += loss.item()

                if self.num_updates is not None and batch_id >= self.num_updates:
                    break

        return model, running_loss / (batch_id + 1)

#seeds
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Global configuration for async experimentation
# -----------------------------------------------------------------------------
RUN_MODE = 'async'  # set to 'async' to enable asynchronous simulation
PAUSE_DURING_UNLEARN = True
STALENESS_LAMBDA = 0.3
SIM_MIN_SLEEP = 0.0
SIM_MAX_SLEEP = 0.2
ASYNC_SNAPSHOT_KEEP = 20
ASYNC_EVAL_INTERVAL = 5
ASYNC_MAX_VERSION = 50
ASYNC_UNLEARN_TRIGGER = 25
BASE_CLIENT_LR = 1e-2
POST_UNLEARN_LR = 1e-2
POST_UNLEARN_MAX_UPDATES = None
ASYNC_STOP_AFTER_UNLEARN_ROUNDS = 40

# -----------------------------------------------------------------------------
# Anchor regularization configuration (only used post-unlearning)
# -----------------------------------------------------------------------------
USE_ANCHOR_REG = True
ANCHOR_BETA = 1e-3

# -----------------------------------------------------------------------------
# Trigger-forgetting loss configuration
# -----------------------------------------------------------------------------
USE_FORGETTING_LOSS = True
GAMMA_FORGET = 1.0
TRIGGER_NUM_SAMPLES = 1000
TRIGGER_BATCH_SIZE = 128
FORGETTING_LR = 5e-3


def clone_state_dict(state_dict):
    return {k: v.clone().detach() for k, v in state_dict.items()}


def subtract_state_dict(state_a, state_b):
    return {k: state_a[k] - state_b[k] for k in state_a}


def build_trigger_loader(x_pool, attack, example_target, num_samples=TRIGGER_NUM_SAMPLES,
                         batch_size=TRIGGER_BATCH_SIZE):
    """Construct a fixed trigger validation loader for forgetting updates."""
    total = len(x_pool)
    if total == 0:
        return None
    num_samples = min(num_samples, total)
    selected_indices = np.random.choice(total, num_samples, replace=False)
    poisoned_data, _ = attack.poison(x_pool[selected_indices], y=example_target, broadcast=True)
    poisoned_data = np.expand_dims(poisoned_data, axis=1)
    trigger_targets = torch.full((num_samples,), int(np.argmax(example_target)), dtype=torch.long)
    dataset = TensorDataset(torch.Tensor(poisoned_data), trigger_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def compute_forgetting_loss(logits, target_label):
    """Penalize the probability of predicting the trigger target label."""
    probs = torch.softmax(logits, dim=1)
    p_target = probs[:, target_label]
    return torch.mean(p_target)


def apply_forgetting_loss_on_model(model, trigger_loader, target_label,
                                   gamma_forget=GAMMA_FORGET,
                                   lr=FORGETTING_LR, max_batches=None):
    if trigger_loader is None:
        return
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for batch_id, (x_batch, _) in enumerate(trigger_loader):
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        forget_loss = compute_forgetting_loss(logits, target_label)
        loss = gamma_forget * forget_loss
        loss.backward()
        optimizer.step()
        if max_batches is not None and (batch_id + 1) >= max_batches:
            break


class FusionAsync:
    """Simple asynchronous fusion applying weighted parameter deltas."""

    def apply_update(self, model, delta_state, weight):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                param.add_(weight * delta_state[name].to(param.device))


class AsyncServer:
    """Server coordinating asynchronous updates and unlearning tasks."""

    def __init__(self, initial_state, num_clients, pause_during_unlearn=True,
                 staleness_lambda=0.1, snapshot_keep=10):
        self.global_model = FLNet().to(device)
        self.global_model.load_state_dict(copy.deepcopy(initial_state))
        self.global_version = 0
        self.total_updates = 0
        self.num_clients = num_clients
        self.pause_during_unlearn = pause_during_unlearn
        self.staleness_lambda = staleness_lambda
        self.fusion = FusionAsync()

        self.client_status = {}
        self.last_client_models = {}
        self.update_queue = queue.Queue()
        self.unlearn_queue = queue.Queue()
        self.snapshots = deque(maxlen=snapshot_keep)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.is_unlearning = False
        self.unlearn_complete_event = threading.Event()
        self.processing_thread = None
        self.trigger_loader = None
        self.trigger_iterator = None
        self.forget_enabled = False
        self.target_label = None
        self.forget_lr = FORGETTING_LR
        self.gamma_forget = GAMMA_FORGET

        self.save_snapshot(self.global_version, copy.deepcopy(self.global_model.state_dict()))

    def start(self):
        if self.processing_thread is not None:
            return
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.processing_thread is not None:
            self.processing_thread.join()
            self.processing_thread = None

    def register_client(self, client_id):
        with self.lock:
            self.client_status[client_id] = {
                'local_version': self.global_version,
                'staleness': 0,
                'active': True
            }
            self.last_client_models[client_id] = copy.deepcopy(self.global_model.state_dict())

    def deregister_client(self, client_id):
        with self.lock:
            if client_id in self.client_status:
                del self.client_status[client_id]
            if client_id in self.last_client_models:
                del self.last_client_models[client_id]
            self.num_clients = max(0, self.num_clients - 1)
            logging.info(f"[SERVER] Client {client_id} deregistered; active clients={self.num_clients}")

    def enqueue_update(self, update):
        self.update_queue.put(update)

    def enqueue_unlearn_result(self, payload):
        self.unlearn_queue.put(payload)

    def get_latest_model(self):
        with self.lock:
            return copy.deepcopy(self.global_model.state_dict()), self.global_version

    def save_snapshot(self, version, state_dict):
        self.snapshots.append((version, copy.deepcopy(state_dict)))
        logging.info(f"[SNAPSHOT] model_ref(v_t={version}) saved")

    def get_snapshot(self, version=None):
        with self.lock:
            if version is None:
                return copy.deepcopy(self.global_model.state_dict()), self.global_version
            for v, state in reversed(self.snapshots):
                if v == version:
                    return copy.deepcopy(state), v
            latest_state, latest_version = self.snapshots[-1]
            return copy.deepcopy(latest_state), latest_version

    def get_client_model(self, client_id):
        with self.lock:
            return copy.deepcopy(self.last_client_models.get(client_id, self.global_model.state_dict()))

    def compute_model_ref(self, client_id, version=None):
        snapshot_state, snapshot_version = self.get_snapshot(version)
        erased_state = self.get_client_model(client_id)

        ref_model = FLNet().to(device)
        ref_model.load_state_dict(copy.deepcopy(snapshot_state))
        erased_model = FLNet().to(device)
        erased_model.load_state_dict(copy.deepcopy(erased_state))

        global_vec = nn.utils.parameters_to_vector(ref_model.parameters())
        erased_vec = nn.utils.parameters_to_vector(erased_model.parameters())

        num_clients = max(2, self.num_clients)
        ref_vec = (num_clients / (num_clients - 1)) * global_vec - (1 / (num_clients - 1)) * erased_vec

        nn.utils.vector_to_parameters(ref_vec, ref_model.parameters())
        logging.info(f"[EVENT] model_ref computed for client {client_id} at version {snapshot_version}")
        return copy.deepcopy(ref_model.state_dict()), snapshot_version

    def begin_unlearning(self):
        self.is_unlearning = True
        self.unlearn_complete_event.clear()
        logging.info(f"[EVENT] Unlearning started (ref v_t={self.global_version})")

    def wait_for_update_queue(self):
        self.update_queue.join()

    def wait_for_unlearn_completion(self):
        self.unlearn_complete_event.wait()

    def _process_loop(self):
        while not self.stop_event.is_set():
            processed = False
            try:
                payload = self.unlearn_queue.get_nowait()
                self._apply_unlearn(payload)
                self.unlearn_queue.task_done()
                processed = True
            except queue.Empty:
                pass

            if processed:
                continue

            if self.is_unlearning:
                time.sleep(0.05)
                continue

            try:
                update = self.update_queue.get(timeout=0.1)
                self._apply_update(update)
                self.update_queue.task_done()
            except queue.Empty:
                continue

    def _apply_update(self, update):
        client_id = update['client_id']
        version_seen = update['version_seen']
        delta_state = update['delta']
        client_state = update['client_state']

        with self.lock:
            staleness = max(0, self.global_version - version_seen)
            weight = math.exp(-self.staleness_lambda * staleness)
            self.fusion.apply_update(self.global_model, delta_state, weight)
            self.global_version += 1
            self.total_updates += 1
            self.client_status[client_id] = {
                'local_version': self.global_version,
                'staleness': staleness,
                'active': True
            }
            self.last_client_models[client_id] = copy.deepcopy(client_state)
            self.save_snapshot(self.global_version, self.global_model.state_dict())
            logging.info(f"[SERVER] version={self.global_version} | updates={self.total_updates} | staleness={staleness} | w={weight:.2f}")

        if self.forget_enabled:
            self._apply_forgetting_step()

    def _apply_unlearn(self, payload):
        state_dict = payload['state_dict']
        mode = payload.get('mode', 'replace')

        with self.lock:
            if mode == 'replace':
                self.global_model.load_state_dict(copy.deepcopy(state_dict))
            else:
                raise ValueError(f"Unsupported unlearn fusion mode: {mode}")

            self.global_version += 1
            self.total_updates += 1
            self.save_snapshot(self.global_version, self.global_model.state_dict())
            logging.info(f"[EVENT] Unlearning done -> global_version={self.global_version} ({mode})")

            for client_id in self.client_status:
                self.client_status[client_id]['local_version'] = self.global_version

            self.is_unlearning = False
            self.unlearn_complete_event.set()

    def configure_trigger_forgetting(self, trigger_loader, target_label, forget_lr=FORGETTING_LR):
        with self.lock:
            self.trigger_loader = trigger_loader
            self.trigger_iterator = None
            self.target_label = target_label
            self.forget_lr = forget_lr
            self.forget_enabled = trigger_loader is not None
        if self.forget_enabled:
            logging.info(f"[SERVER] Trigger forgetting enabled (gamma={self.gamma_forget}, lr={forget_lr})")
        else:
            logging.info("[SERVER] Trigger forgetting disabled")

    def _apply_forgetting_step(self):
        with self.lock:
            if not self.forget_enabled or self.trigger_loader is None or self.target_label is None:
                return
            if self.trigger_iterator is None:
                self.trigger_iterator = iter(self.trigger_loader)
            try:
                batch = next(self.trigger_iterator)
            except StopIteration:
                self.trigger_iterator = iter(self.trigger_loader)
                batch = next(self.trigger_iterator)
            x_batch, _ = batch
            x_batch = x_batch.to(device)
            optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.forget_lr, momentum=0.9)
            optimizer.zero_grad()
            logits = self.global_model(x_batch)
            forget_loss = compute_forgetting_loss(logits, self.target_label)
            loss = self.gamma_forget * forget_loss
            loss.backward()
            optimizer.step()
        logging.info(f"[SERVER-FORGET] loss={forget_loss.item():.4f}")


class ClientSimulator:
    def __init__(self, client_id, trainloader, server, num_local_epochs=1,
                 num_updates_in_epoch=None, compute_speed=(0.0, 0.2),
                 base_lr=BASE_CLIENT_LR):
        self.client_id = client_id
        self.trainloader = trainloader
        self.server = server
        self.compute_speed = compute_speed
        self.local_version = 0
        self.active_event = threading.Event()
        self.active_event.set()
        self.stop_event = threading.Event()
        self.thread = None
        self.config_lock = threading.Lock()
        self.current_lr = base_lr
        self.set_local_training_config(num_local_epochs=num_local_epochs,
                                       num_updates_in_epoch=num_updates_in_epoch)
        self.anchor_state_dict = None
        self.use_anchor_reg = False
        self.anchor_beta = 0.0

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def pause(self):
        self.active_event.clear()

    def resume(self):
        self.active_event.set()

    def set_lr(self, new_lr):
        with self.config_lock:
            self.current_lr = new_lr
        logging.info(f"[CLIENT {self.client_id}] learning_rate set to {new_lr}")

    def set_anchor_regularization(self, anchor_state_dict=None, use_anchor_reg=False,
                                   anchor_beta=0.0):
        with self.config_lock:
            self.anchor_state_dict = copy.deepcopy(anchor_state_dict) if anchor_state_dict is not None else None
            self.use_anchor_reg = use_anchor_reg and anchor_state_dict is not None
            self.anchor_beta = anchor_beta
        if self.use_anchor_reg:
            logging.info(f"[CLIENT {self.client_id}] Anchor regularization enabled (beta={anchor_beta})")
        else:
            logging.info(f"[CLIENT {self.client_id}] Anchor regularization disabled")

    def set_local_training_config(self, num_local_epochs=1, num_updates_in_epoch=None):
        with self.config_lock:
            self.local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch,
                                                num_local_epochs=num_local_epochs)
        logging.info(f"[CLIENT {self.client_id}] local training config updated: epochs={num_local_epochs}, "
                     f"max_updates={num_updates_in_epoch}")

    def _run(self):
        while not self.stop_event.is_set():
            if not self.active_event.is_set():
                time.sleep(0.05)
                continue

            global_state, global_version = self.server.get_latest_model()
            model = FLNet().to(device)
            model.load_state_dict(copy.deepcopy(global_state))
            self.local_version = global_version

            initial_state = copy.deepcopy(model.state_dict())
            with self.config_lock:
                training_runner = self.local_training
                lr = self.current_lr
                anchor_state = copy.deepcopy(self.anchor_state_dict) if self.anchor_state_dict is not None else None
                use_anchor = self.use_anchor_reg
                anchor_beta = self.anchor_beta

            model_update, loss = training_runner.train(model=model,
                                                        trainloader=self.trainloader,
                                                        criterion=None, opt=None,
                                                        lr=lr,
                                                        anchor_state_dict=anchor_state,
                                                        use_anchor_reg=use_anchor,
                                                        anchor_beta=anchor_beta)
            updated_state = copy.deepcopy(model_update.state_dict())
            delta_state = subtract_state_dict(updated_state, initial_state)

            update_payload = {
                'client_id': self.client_id,
                'delta': clone_state_dict(delta_state),
                'client_state': copy.deepcopy(updated_state),
                'version_seen': global_version,
                'flag': 'update',
                'timestamp': time.time()
            }

            self.server.enqueue_update(update_payload)
            delay = random.uniform(self.compute_speed[0], self.compute_speed[1])
            logging.info(f"[CLIENT {self.client_id}] local_version={self.local_version} | delay={delay:.2f}s")
            time.sleep(delay)


class UnlearnWorker(threading.Thread):
    def __init__(self, server, target_client_id, trainloader, num_local_epochs=2,
                 lr=0.001, clip_grad=5, mode='replace', reference_version=None,
                 testloader=None, testloader_poison=None):
        super().__init__(daemon=True)
        self.server = server
        self.target_client_id = target_client_id
        self.trainloader = trainloader
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.clip_grad = clip_grad
        self.mode = mode
        self.reference_version = reference_version
        self.testloader = testloader
        self.testloader_poison = testloader_poison

    def run(self):
        model_ref_state, ref_version = self.server.compute_model_ref(self.target_client_id,
                                                                     version=self.reference_version)
        model_ref = FLNet().to(device)
        model_ref.load_state_dict(copy.deepcopy(model_ref_state))

        erased_model_state = self.server.get_client_model(self.target_client_id)
        erased_model = FLNet().to(device)
        erased_model.load_state_dict(copy.deepcopy(erased_model_state))

        if self.testloader is not None and self.testloader_poison is not None:
            eval_model = copy.deepcopy(model_ref)
            unlearn_clean_acc = Utils.evaluate(self.testloader, eval_model)
            logging.info(f'Clean Accuracy for Reference Model = {unlearn_clean_acc}')
            unlearn_pois_acc = Utils.evaluate(self.testloader_poison, eval_model)
            logging.info(f'Backdoor Accuracy for Reference Model = {unlearn_pois_acc}')

        dist_ref_random_lst = []
        for _ in range(10):
            dist_ref_random_lst.append(Utils.get_distance(model_ref, FLNet().to(device)))
        threshold = np.mean(dist_ref_random_lst) / 9
        radius = math.sqrt(threshold)
        dist_ref_party = Utils.get_distance(model_ref, erased_model)
        logging.info(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
        logging.info(f'Radius for model_ref (squared distance): {threshold} | sqrt={radius}')
        logging.info(f'Distance of Reference Model to party0_model: {dist_ref_party}')

        model = copy.deepcopy(model_ref)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        model.train()
        for epoch in range(self.num_local_epochs):
            logging.info(f'------------ {epoch}')
            for batch_id, (x_batch, y_batch) in enumerate(self.trainloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                opt.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss_joint = -loss
                loss_joint.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

                opt.step()

                with torch.no_grad():
                    distance = Utils.get_distance(model, model_ref)
                    if distance > threshold:
                        dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                        dist_vec = dist_vec/torch.norm(dist_vec)*radius
                        proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        nn.utils.vector_to_parameters(proj_vec, model.parameters())

                distance_ref_party_0 = Utils.get_distance(model, erased_model)
                if batch_id % 20 == 0:
                    logging.info(f'Distance from unlearned model to party 0: {distance_ref_party_0}; to ref={distance}')

        unlearned_model_state = copy.deepcopy(model.state_dict())
        self.server.enqueue_unlearn_result({
            'state_dict': unlearned_model_state,
            'mode': self.mode,
            'reference_version': ref_version
        })

def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):

    fusion_class_dict = {
        'FedAvg': FusionAvg(num_parties),
        'Retrain': FusionRetrain(num_parties), 
        'Unlearn': FusionAvg(num_parties)
        }

    return fusion_class_dict[fusion_key]

num_parties = 5
scale = 1
# Currently, we assume that the party to be erased is party_id = 0
party_to_be_erased = 0
num_samples_erased_party = int(60000 / num_parties * scale)
num_samples_per_party = int((60000 - num_samples_erased_party)/(num_parties - 1))
print('Number of samples erased party:', num_samples_erased_party)
print('Number of samples other party:', num_samples_per_party)

(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)

x_train, y_train = preprocess(x_raw, y_raw)
x_test, y_test = preprocess(x_raw_test, y_raw_test)

n_train = np.shape(y_train)[0]
shuffled_indices = np.arange(n_train)
np.random.shuffle(shuffled_indices)
x_train = x_train[shuffled_indices]
y_train = y_train[shuffled_indices]

x_train_party = x_train[0:num_samples_erased_party]
y_train_party = y_train[0:num_samples_erased_party]

backdoor = PoisoningAttackBackdoor(add_pattern_bd)
example_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
target_label = int(np.argmax(example_target))

percent_poison = .8

all_indices = np.arange(len(x_train_party))
remove_indices = all_indices[np.all(y_train_party == example_target, axis=1)]

target_indices = list(set(all_indices) - set(remove_indices))
num_poison = int(percent_poison * len(target_indices))
print(f'num poison: {num_poison}')
selected_indices = np.random.choice(target_indices, num_poison, replace=False)

poisoned_data, poisoned_labels = backdoor.poison(x_train_party[selected_indices], y=example_target, broadcast=True)

poisoned_x_train = np.copy(x_train_party)
poisoned_y_train = np.argmax(y_train_party,axis=1)
for s,i in zip(selected_indices,range(len(selected_indices))):
    poisoned_x_train[s] = poisoned_data[i]
    poisoned_y_train[s] = int(np.argmax(poisoned_labels[i]))

plt.imshow(poisoned_x_train[selected_indices[0]])
print(poisoned_y_train[0])

poisoned_x_train_ch = np.expand_dims(poisoned_x_train, axis = 1)
print('poisoned_x_train_ch.shape:',poisoned_x_train_ch.shape)
print('poisoned_y_train.shape:',poisoned_y_train.shape)
poisoned_dataset_train = TensorDataset(torch.Tensor(poisoned_x_train_ch),torch.Tensor(poisoned_y_train).long())
poisoned_dataloader_train = DataLoader(poisoned_dataset_train, batch_size=128, shuffle=True)

num_samples = (num_parties - 1) * num_samples_per_party
x_train_parties = x_train[num_samples_erased_party:num_samples_erased_party+num_samples] 
x_train_parties_ch = np.expand_dims(x_train_parties, axis=1)
y_train_parties = y_train[num_samples_erased_party:num_samples_erased_party+num_samples]
y_train_parties_c = np.argmax(y_train_parties, axis=1).astype(int)
print(x_train_parties_ch.shape)
print(y_train_parties_c.shape)

x_train_parties = TensorDataset(torch.Tensor(x_train_parties_ch), torch.Tensor(y_train_parties_c).long())
clean_dataset_train = torch.utils.data.random_split(x_train_parties, [num_samples_per_party for _ in range(1, num_parties)])

trainloader_lst = [poisoned_dataloader_train] 
for i in range(len(clean_dataset_train)):
    trainloader_lst.append(DataLoader(clean_dataset_train[i], batch_size=128, shuffle=True))

all_indices = np.arange(len(x_test))
remove_indices = all_indices[np.all(y_test == example_target, axis=1)]

target_indices = list(set(all_indices) - set(remove_indices))
print('num poison test:', len(target_indices))
poisoned_data, poisoned_labels = backdoor.poison(x_test[target_indices], y=example_target, broadcast=True)

poisoned_x_test = np.copy(x_test)
poisoned_y_test = np.argmax(y_test,axis=1)

for s,i in zip(target_indices,range(len(target_indices))):
    poisoned_x_test[s] = poisoned_data[i]
    poisoned_y_test[s] = int(np.argmax(poisoned_labels[i]))

poisoned_x_test_ch = np.expand_dims(poisoned_x_test, axis = 1)
print(poisoned_x_test_ch.shape)
print(poisoned_y_test.shape)
poisoned_dataset_test = TensorDataset(torch.Tensor(poisoned_x_test_ch),torch.Tensor(poisoned_y_test).long())
testloader_poison = DataLoader(poisoned_dataset_test, batch_size=1000, shuffle=False)

x_test_pt = np.expand_dims(x_test, axis = 1)
y_test_pt = np.argmax(y_test,axis=1).astype(int)
print(x_test_pt.shape)
print(y_test_pt.shape)
dataset_test = TensorDataset(torch.Tensor(x_test_pt), torch.Tensor(y_test_pt).long())
testloader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

trigger_loader = None
if USE_FORGETTING_LOSS:
    trigger_loader = build_trigger_loader(x_test, backdoor, example_target,
                                          num_samples=TRIGGER_NUM_SAMPLES,
                                          batch_size=TRIGGER_BATCH_SIZE)
    logging.info(f"[TRIGGER] Built trigger validation loader with {TRIGGER_NUM_SAMPLES} samples")

if RUN_MODE == 'sync':
    num_of_repeats = 1
    num_fl_rounds = 10

    fusion_types = ['FedAvg', 'Retrain']
    fusion_types_unlearn = ['Retrain', 'Unlearn']

    num_updates_in_epoch = None
    num_local_epochs = 1

    dist_Retrain = {}
    loss_fed = {}
    clean_accuracy = {}
    pois_accuracy = {}
    for fusion_key in fusion_types:
        loss_fed[fusion_key] = np.zeros(num_fl_rounds)
        clean_accuracy[fusion_key] = np.zeros(num_fl_rounds)
        pois_accuracy[fusion_key] = np.zeros(num_fl_rounds)
        if fusion_key != 'Retrain':
            dist_Retrain[fusion_key] = np.zeros(num_fl_rounds)

    party_models_dict = {}

    initial_model = FLNet().to(device)
    model_dict = {}
    for fusion_key in fusion_types:
        model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

    for round_num in range(num_fl_rounds):
        local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

        for fusion_key in fusion_types:
            fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)

            current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
            current_model = copy.deepcopy(initial_model)
            current_model.load_state_dict(current_model_state_dict)

            ##################### Local Training Round #############################
            party_models = []
            party_losses = []
            for party_id in range(num_parties):

                if fusion_key == 'Retrain' and party_id == party_to_be_erased:
                    party_models.append(FLNet().to(device))
                else:
                    model = copy.deepcopy(current_model)
                    model_update, party_loss = local_training.train(model=model,
                                                trainloader=trainloader_lst[party_id],
                                                criterion=None, opt=None)

                    party_models.append(copy.deepcopy(model_update))
                    party_losses.append(party_loss)

            loss_fed[fusion_key][round_num] += (np.mean(party_losses)/num_of_repeats)
            ######################################################################

            current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)

            model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
            party_models_dict[fusion_key] = party_models

            eval_model = FLNet().to(device)
            eval_model.load_state_dict(current_model_state_dict)
            clean_acc = Utils.evaluate(testloader, eval_model)
            clean_accuracy[fusion_key][round_num] = clean_acc
            print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {clean_acc}')
            pois_acc = Utils.evaluate(testloader_poison, eval_model)
            pois_accuracy[fusion_key][round_num] = pois_acc
            print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {pois_acc}')

    for fusion_key in fusion_types:
        current_model_state_dict = model_dict[fusion_key]
        current_model = copy.deepcopy(initial_model)
        current_model.load_state_dict(current_model_state_dict)
        clean_acc = Utils.evaluate(testloader, current_model)
        print(f'Clean Accuracy {fusion_key}: {clean_acc}')
        pois_acc = Utils.evaluate(testloader_poison, current_model)
        print(f'Backdoor Accuracy {fusion_key}: {pois_acc}')

    num_updates_in_epoch = None
    num_local_epochs_unlearn = 5
    lr = 0.01
    clip_grad = 5


    initial_model = FLNet().to(device)
    unlearned_model_dict = {}
    for fusion_key in fusion_types_unlearn:
        if fusion_key == 'Retrain':
            unlearned_model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

    clean_accuracy_unlearn = {}
    pois_accuracy_unlearn = {}
    for fusion_key in fusion_types_unlearn:
        clean_accuracy_unlearn[fusion_key] = 0
        pois_accuracy_unlearn[fusion_key] = 0

    anchor_state_post_unlearn = None
    for fusion_key in fusion_types:
        if fusion_key == 'Retrain':
            continue

        initial_model = FLNet().to(device)
        fedavg_model_state_dict = copy.deepcopy(model_dict[fusion_key])
        fedavg_model = copy.deepcopy(initial_model)
        fedavg_model.load_state_dict(fedavg_model_state_dict)

        party_models = copy.deepcopy(party_models_dict[fusion_key])
        party0_model = copy.deepcopy(party_models[0])

        #compute reference model
        #w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}
        model_ref_vec = num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters()) \
                                   - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(party0_model.parameters())

        #compute threshold
        model_ref = copy.deepcopy(initial_model)
        nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

        eval_model = copy.deepcopy(model_ref)
        unlearn_clean_acc = Utils.evaluate(testloader, eval_model)
        print(f'Clean Accuracy for Reference Model = {unlearn_clean_acc}')
        unlearn_pois_acc = Utils.evaluate(testloader_poison, eval_model)
        print(f'Backdoor Accuracy for Reference Model = {unlearn_pois_acc}')

        dist_ref_random_lst = []
        for _ in range(10):
            dist_ref_random_lst.append(Utils.get_distance(model_ref, FLNet().to(device)))

        print(f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
        threshold = np.mean(dist_ref_random_lst) / 3
        print(f'Radius for model_ref: {threshold}')
        dist_ref_party = Utils.get_distance(model_ref, party0_model)
        print(f'Distance of Reference Model to party0_model: {dist_ref_party}')


        ###############################################################
        #### Unlearning
        ###############################################################
        model = copy.deepcopy(model_ref)

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        model.train()
        for epoch in range(num_local_epochs_unlearn):
            print('------------', epoch)
            for batch_id, (x_batch, y_batch) in enumerate(trainloader_lst[party_to_be_erased]):

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                opt.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss_joint = -loss # negate the loss for gradient ascent
                loss_joint.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                opt.step()

                with torch.no_grad():
                    distance = Utils.get_distance(model, model_ref)
                    if distance > threshold:
                        dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                        dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(threshold)
                        proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        nn.utils.vector_to_parameters(proj_vec, model.parameters())
                        distance = Utils.get_distance(model, model_ref)

                distance_ref_party_0 = Utils.get_distance(model, party0_model)
                print('Distance from the unlearned model to party 0:', distance_ref_party_0.item())

                if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
                    break
        ####################################################################

        unlearned_model = copy.deepcopy(model)
        unlearned_model_dict[fusion_types_unlearn[1]] = unlearned_model.state_dict()
        if USE_ANCHOR_REG:
            anchor_state_post_unlearn = copy.deepcopy(unlearned_model_dict[fusion_types_unlearn[1]])

        eval_model = FLNet().to(device)
        eval_model.load_state_dict(unlearned_model_dict[fusion_types_unlearn[1]])
        unlearn_clean_acc = Utils.evaluate(testloader, eval_model)
        print(f'Clean Accuracy for UN-Local Model = {unlearn_clean_acc}')
        clean_accuracy_unlearn[fusion_types_unlearn[1]] =  unlearn_clean_acc
        pois_unlearn_acc = Utils.evaluate(testloader_poison, eval_model)
        print(f'Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}')
        pois_accuracy_unlearn[fusion_types_unlearn[1]] =  pois_unlearn_acc

    num_fl_after_unlearn_rounds = num_fl_rounds
    num_updates_in_epoch = 50
    num_local_epochs = 1

    clean_accuracy_unlearn_fl_after_unlearn = {}
    pois_accuracy_unlearn_fl_after_unlearn = {}
    loss_unlearn = {}
    for fusion_key in fusion_types_unlearn:
        clean_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
        pois_accuracy_unlearn_fl_after_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)
        loss_unlearn[fusion_key] = np.zeros(num_fl_after_unlearn_rounds)


    for round_num in range(num_fl_after_unlearn_rounds):

        local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

        for fusion_key in fusion_types_unlearn:
            # Reduce num_parties by 1 to remove the erased party
            fusion = FL_round_fusion_selection(num_parties=num_parties - 1, fusion_key=fusion_key)

            current_model_state_dict = copy.deepcopy(unlearned_model_dict[fusion_key])
            current_model = FLNet().to(device)
            current_model.load_state_dict(current_model_state_dict)

            ##################### Local Training Round #############################
            party_models = []
            party_losses = []
            anchor_state_for_round = None
            if USE_ANCHOR_REG and fusion_key == 'Unlearn' and anchor_state_post_unlearn is not None:
                anchor_state_for_round = copy.deepcopy(anchor_state_post_unlearn)
            for party_id in range(1, num_parties):
                model = copy.deepcopy(current_model)
                model_update, party_loss = local_training.train(model=model,
                                            trainloader=trainloader_lst[party_id],
                                            criterion=None, opt=None,
                                            anchor_state_dict=anchor_state_for_round,
                                            use_anchor_reg=(anchor_state_for_round is not None),
                                            anchor_beta=ANCHOR_BETA)

                party_models.append(copy.deepcopy(model_update))
                party_losses.append(party_loss)

            loss_unlearn[fusion_key][round_num] = np.mean(party_losses)
            ######################################################################

            current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)
            if fusion_key == 'Unlearn' and USE_FORGETTING_LOSS and trigger_loader is not None:
                forgetting_model = FLNet().to(device)
                forgetting_model.load_state_dict(copy.deepcopy(current_model_state_dict))
                apply_forgetting_loss_on_model(forgetting_model, trigger_loader,
                                               target_label,
                                               gamma_forget=GAMMA_FORGET,
                                               lr=FORGETTING_LR)
                current_model_state_dict = copy.deepcopy(forgetting_model.state_dict())
            unlearned_model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
            party_models_dict[fusion_key] = party_models

            eval_model = FLNet().to(device)
            eval_model.load_state_dict(current_model_state_dict)
            unlearn_clean_acc = Utils.evaluate(testloader, eval_model)
            print(f'Global Clean Accuracy {fusion_key}, round {round_num} = {unlearn_clean_acc}')
            clean_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_clean_acc
            unlearn_pois_acc = Utils.evaluate(testloader_poison, eval_model)
            print(f'Global Backdoor Accuracy {fusion_key}, round {round_num} = {unlearn_pois_acc}')
            pois_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_pois_acc


    fl_rounds = [i for i in range(1, num_fl_rounds + 1)]

    plt.plot(fl_rounds, clean_accuracy_unlearn_fl_after_unlearn['Unlearn'], 'ro--', linewidth=2, markersize=12, label='UN-Clean Acc')
    plt.plot(fl_rounds, pois_accuracy_unlearn_fl_after_unlearn['Unlearn'], 'gx--', linewidth=2, markersize=12, label='UN-Backdoor Acc')
    plt.plot(fl_rounds, clean_accuracy_unlearn_fl_after_unlearn['Retrain'], 'm^-', linewidth=2, markersize=12, label='Retrain-Clean Acc')
    plt.plot(fl_rounds, pois_accuracy_unlearn_fl_after_unlearn['Retrain'], 'c+-', linewidth=2, markersize=12, label='Retrain-Backdoor Acc')
    plt.xlabel('Training Rounds')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.ylim([0, 100])
    plt.xlim([1, 10])
    plt.legend()
    plt.show()
else:
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'federated-unlearning/doc/logs/async_unlearning_log_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    async_initial_model = FLNet().to(device)
    server = AsyncServer(async_initial_model.state_dict(), num_parties,
                         pause_during_unlearn=PAUSE_DURING_UNLEARN,
                         staleness_lambda=STALENESS_LAMBDA,
                         snapshot_keep=ASYNC_SNAPSHOT_KEEP)
    server.start()

    clients = []
    speed_factors = np.linspace(1.0, 1.0 + 0.5 * (num_parties - 1), num_parties)
    for client_id in range(num_parties):
        server.register_client(client_id)
        factor = speed_factors[client_id]
        min_delay = SIM_MIN_SLEEP * factor
        max_delay = max(min_delay + 0.01, SIM_MAX_SLEEP * factor + 0.01 * client_id)
        client = ClientSimulator(client_id=client_id,
                                 trainloader=trainloader_lst[client_id],
                                 server=server,
                                 num_local_epochs=1,
                                 num_updates_in_epoch=None,
                                 compute_speed=(min_delay, max_delay),
                                 base_lr=BASE_CLIENT_LR)
        clients.append(client)
        client.start()
        logging.info(f"[CLIENT {client_id}] compute_speed=({min_delay:.2f}, {max_delay:.2f})")

    eval_versions = []
    clean_history = []
    poison_history = []
    recorded_versions = set()

    init_state, init_version = server.get_snapshot()
    eval_model = FLNet().to(device)
    eval_model.load_state_dict(init_state)
    init_clean = Utils.evaluate(testloader, eval_model)
    init_pois = Utils.evaluate(testloader_poison, eval_model)
    logging.info(f"[ASYNC-EVAL] version={init_version} | Clean={init_clean} | Backdoor={init_pois}")
    eval_versions.append(init_version)
    clean_history.append(init_clean)
    poison_history.append(init_pois)
    recorded_versions.add(init_version)

    unlearn_triggered = False
    reference_version = None
    erased_client_removed = False
    post_unlearn_stop_version = None
    post_unlearn_adjustment_done = False

    try:
        while server.global_version < ASYNC_MAX_VERSION:
            time.sleep(0.5)
            current_version = server.global_version

            if (not unlearn_triggered) and current_version >= ASYNC_UNLEARN_TRIGGER:
                if PAUSE_DURING_UNLEARN:
                    for client in clients:
                        client.pause()
                    server.wait_for_update_queue()

                reference_version = server.global_version
                server.begin_unlearning()

                worker = UnlearnWorker(server=server,
                                       target_client_id=party_to_be_erased,
                                       trainloader=trainloader_lst[party_to_be_erased],
                                       num_local_epochs=2,
                                       lr=0.001,
                                       clip_grad=5,
                                       mode='replace',
                                       reference_version=reference_version,
                                       testloader=testloader,
                                       testloader_poison=testloader_poison)
                worker.start()
                worker.join()
                server.wait_for_unlearn_completion()

                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = FLNet().to(device)
                eval_model.load_state_dict(snapshot_state)
                post_clean = Utils.evaluate(testloader, eval_model)
                post_pois = Utils.evaluate(testloader_poison, eval_model)
                logging.info(f"[POST-UNLEARN-EVAL] version={snapshot_version} | Clean={post_clean} | Backdoor={post_pois}")
                eval_versions.append(snapshot_version)
                clean_history.append(post_clean)
                poison_history.append(post_pois)
                recorded_versions.add(snapshot_version)
                post_unlearn_stop_version = snapshot_version + ASYNC_STOP_AFTER_UNLEARN_ROUNDS
                logging.info(f"[SERVER] Post-unlearn evaluations logged; async stop target version set to {post_unlearn_stop_version}")

                if USE_ANCHOR_REG:
                    for client in clients:
                        client.set_anchor_regularization(anchor_state_dict=snapshot_state,
                                                         use_anchor_reg=True,
                                                         anchor_beta=ANCHOR_BETA)
                    logging.info("[SERVER] Anchor regularization enabled for surviving clients")

                if USE_FORGETTING_LOSS and trigger_loader is not None:
                    server.configure_trigger_forgetting(trigger_loader, target_label,
                                                         forget_lr=FORGETTING_LR)

                if not erased_client_removed:
                    erased_index = None
                    for idx, client in enumerate(clients):
                        if client.client_id == party_to_be_erased:
                            client.stop()
                            erased_index = idx
                            break
                    if erased_index is not None:
                        clients.pop(erased_index)
                        server.deregister_client(party_to_be_erased)
                        erased_client_removed = True
                        logging.info(f"[SERVER] Client {party_to_be_erased} removed post-unlearning")
                    else:
                        logging.warning(f"[SERVER] Target client {party_to_be_erased} not found during removal")

                if PAUSE_DURING_UNLEARN:
                    for client in clients:
                        client.resume()

                if not post_unlearn_adjustment_done:
                    for client in clients:
                        client.set_lr(POST_UNLEARN_LR)
                        client.set_local_training_config(num_local_epochs=1,
                                                         num_updates_in_epoch=POST_UNLEARN_MAX_UPDATES)
                    post_unlearn_adjustment_done = True
                    logging.info(f"[SERVER] Clients switched to post-unlearn LR={POST_UNLEARN_LR} and max_updates={POST_UNLEARN_MAX_UPDATES}")

                logging.info("[SERVER] Resumed async updates")
                unlearn_triggered = True

            if current_version % ASYNC_EVAL_INTERVAL == 0 and current_version not in recorded_versions:
                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = FLNet().to(device)
                eval_model.load_state_dict(snapshot_state)
                clean_acc = Utils.evaluate(testloader, eval_model)
                pois_acc = Utils.evaluate(testloader_poison, eval_model)
                logging.info(f"[ASYNC-EVAL] version={snapshot_version} | Clean={clean_acc} | Backdoor={pois_acc}")
                eval_versions.append(snapshot_version)
                clean_history.append(clean_acc)
                poison_history.append(pois_acc)
                recorded_versions.add(snapshot_version)

            if (unlearn_triggered and post_unlearn_stop_version is not None and
                    current_version >= post_unlearn_stop_version):
                logging.info(f"[SERVER] Reached post-unlearn stop version {post_unlearn_stop_version}; terminating training loop")
                break

    finally:
        for client in clients:
            client.stop()
        server.wait_for_update_queue()
        server.stop()

    plt.figure()
    plt.plot(eval_versions, clean_history, 'r-', linewidth=2, marker='o', label='Clean Accuracy')
    plt.plot(eval_versions, poison_history, 'b-', linewidth=2, marker='x', label='Backdoor Accuracy')
    if unlearn_triggered and reference_version is not None:
        plt.axvline(x=reference_version + 1, linestyle='--', color='k', label='Unlearn Event')
    plt.xlabel('Global Version')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    
    plot_filename = f'federated-unlearning/doc/images/async_unlearning_plot_{timestamp}.png'
    plt.savefig(plot_filename)
    logging.info(f"Plot saved to {plot_filename}")
    
    plt.show()

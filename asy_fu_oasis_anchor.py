import copy
import logging
import math
import queue
import random
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt

from utils.model import OASISNet
from utils.local_train import LocalTraining as BaseLocalTraining
from utils.utils import Utils


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


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RUN_MODE = 'async'
PAUSE_DURING_UNLEARN = True
STALENESS_LAMBDA = 0.3
SIM_MIN_SLEEP = 0.0
SIM_MAX_SLEEP = 0.2
ASYNC_SNAPSHOT_KEEP = 20
ASYNC_EVAL_INTERVAL = 5
ASYNC_MAX_VERSION = 50
ASYNC_UNLEARN_TRIGGER = 25
BASE_CLIENT_LR = 5e-3
POST_UNLEARN_LR = 5e-3
POST_UNLEARN_MAX_UPDATES = None
ASYNC_STOP_AFTER_UNLEARN_ROUNDS = 40

USE_ANCHOR_REG = True
ANCHOR_BETA = 1e-3


def clone_state_dict(state_dict):
    return {k: v.clone().detach() for k, v in state_dict.items()}


def subtract_state_dict(state_a, state_b):
    return {k: state_a[k] - state_b[k] for k in state_a}


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
        self.global_model = OASISNet().to(device)
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

        ref_model = OASISNet().to(device)
        ref_model.load_state_dict(copy.deepcopy(snapshot_state))
        erased_model = OASISNet().to(device)
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
            try:
                update_payload = self.update_queue.get(timeout=0.25)
            except queue.Empty:
                self._process_unlearn_queue()
                continue

            if update_payload['flag'] == 'update':
                self._apply_client_update(update_payload)
                self.update_queue.task_done()
            elif update_payload['flag'] == 'unlearn':
                self._process_unlearn_queue()
                self.update_queue.task_done()
            else:
                self.update_queue.task_done()

    def _apply_client_update(self, payload):
        client_id = payload['client_id']
        delta_state = payload['delta']
        version_seen = payload['version_seen']

        with self.lock:
            staleness = self.global_version - version_seen
            staleness = max(0, staleness)
            weight = math.exp(-self.staleness_lambda * staleness)
            if staleness > 0:
                logging.info(f"[SERVER] Applying stale update from client {client_id} (staleness={staleness})")

            self.fusion.apply_update(self.global_model, delta_state, weight)
            self.global_version += 1
            self.total_updates += 1
            self.save_snapshot(self.global_version, self.global_model.state_dict())
            self.last_client_models[client_id] = payload['client_state']
            self.client_status[client_id]['local_version'] = self.global_version

        if self.global_version % ASYNC_EVAL_INTERVAL == 0:
            logging.info(f"[SERVER] global_version={self.global_version} total_updates={self.total_updates}")

    def _process_unlearn_queue(self):
        if self.unlearn_queue.empty():
            return
        payload = self.unlearn_queue.get()
        mode = payload.get('mode', 'replace')
        state_dict = payload['state_dict']
        reference_version = payload.get('reference_version')

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


class AsyncClient:
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
            model = OASISNet().to(device)
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


class AsyncUnlearnWorker(threading.Thread):
    def __init__(self, server, target_client_id, trainloader,
                 num_local_epochs=1, lr=0.001, clip_grad=5, mode='replace', reference_version=None,
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
        model_ref = OASISNet().to(device)
        model_ref.load_state_dict(copy.deepcopy(model_ref_state))

        erased_model_state = self.server.get_client_model(self.target_client_id)
        erased_model = OASISNet().to(device)
        erased_model.load_state_dict(copy.deepcopy(erased_model_state))

        if self.testloader is not None and self.testloader_poison is not None:
            eval_model = copy.deepcopy(model_ref)
            unlearn_clean_acc = Utils.evaluate(self.testloader, eval_model)
            logging.info(f'Clean Accuracy for Reference Model = {unlearn_clean_acc}')
            unlearn_pois_acc = Utils.evaluate(self.testloader_poison, eval_model)
            logging.info(f'Backdoor Accuracy for Reference Model = {unlearn_pois_acc}')

        dist_ref_random_lst = []
        for _ in range(10):
            dist_ref_random_lst.append(Utils.get_distance(model_ref, OASISNet().to(device)))
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


def add_oasis_trigger(img_tensor: torch.Tensor) -> torch.Tensor:
    """Add a visible L-shaped trigger patch to a 3×H×W tensor in [0, 1]."""

    img = img_tensor.clone()
    h, w = img.shape[1], img.shape[2]
    patch_size = 16
    y_start = h - patch_size
    x_start = w - patch_size

    color = torch.tensor([1.0, 0.0, 0.0], device=img.device).view(3, 1, 1)
    img[:, y_start:, x_start:x_start + 3] = color
    img[:, y_start:y_start + 3, x_start:] = color
    return img


class OASISAugmentedDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform is not None:
            img = transforms.ToPILImage()(x)
            img = self.transform(img)
            x = img
        return x, y


def _load_balanced_oasis_images(base_dir: Path, max_per_class: int, image_size: int):
    class_names = [
        "Non Demented",
        "Very mild Dementia",
        "Mild Dementia",
        "Moderate Dementia",
    ]
    balanced_paths = []
    balanced_labels = []

    for label, cls_name in enumerate(class_names):
        cls_dir = base_dir / cls_name
        paths = list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.png"))
        random.shuffle(paths)
        selected = paths[: min(len(paths), max_per_class)]
        balanced_paths.extend(selected)
        balanced_labels.extend([label for _ in selected])

    images = []
    labels = []
    for path, label in zip(balanced_paths, balanced_labels):
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size))
            arr = np.array(img)
            if arr.shape != (image_size, image_size, 3):
                continue
            images.append(arr)
            labels.append(label)

    images = np.stack(images).astype(np.float32) / 255.0
    images = np.transpose(images, (0, 3, 1, 2))
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def _dirichlet_split_indices(y_train: np.ndarray, num_parties: int, alpha: float):
    num_classes = len(np.unique(y_train))
    indices_per_party = [[] for _ in range(num_parties)]
    for cls in range(num_classes):
        cls_indices = np.where(y_train == cls)[0]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(alpha=[alpha] * num_parties)
        splits = np.random.multinomial(len(cls_indices), proportions)
        start = 0
        for party_id, count in enumerate(splits):
            end = start + count
            indices_per_party[party_id].extend(cls_indices[start:end].tolist())
            start = end
    for party_id in range(num_parties):
        np.random.shuffle(indices_per_party[party_id])
    return indices_per_party


def _make_party_loader(
    x_party,
    y_party,
    batch_size,
    target_label,
    party_id,
    party_to_be_erased,
    poison_ratio,
    train_transform,
):
    x_tensor = torch.tensor(x_party, dtype=torch.float32)
    y_tensor = torch.tensor(y_party, dtype=torch.long)

    if party_id == party_to_be_erased:
        num_poison = int(len(x_tensor) * poison_ratio)
        poison_indices = np.random.choice(len(x_tensor), num_poison, replace=False)
        for idx in poison_indices:
            x_tensor[idx] = add_oasis_trigger(x_tensor[idx])
            y_tensor[idx] = target_label

    class_counts = torch.bincount(y_tensor, minlength=4).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[y_tensor]
    sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=len(sample_weights), replacement=True)

    dataset = OASISAugmentedDataset(x_tensor, y_tensor, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def make_oasis_federated_loaders(
    base_dir: Path,
    num_parties: int,
    image_size: int = 128,
    max_per_class: int = 2000,
    alpha: float = 1.0,
    batch_size: int = 32,
    target_label: int = 0,
    party_to_be_erased: int = 0,
    poison_ratio: float = 0.6,
):
    images, labels = _load_balanced_oasis_images(
        base_dir=base_dir, max_per_class=max_per_class, image_size=image_size
    )

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.15, stratify=labels, random_state=0
    )

    indices_per_party = _dirichlet_split_indices(y_train, num_parties=num_parties, alpha=alpha)

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )

    trainloader_lst = []
    for party_id, indices in enumerate(indices_per_party):
        x_party = x_train[indices]
        y_party = y_train[indices]
        loader = _make_party_loader(
            x_party=x_party,
            y_party=y_party,
            batch_size=batch_size,
            target_label=target_label,
            party_id=party_id,
            party_to_be_erased=party_to_be_erased,
            poison_ratio=poison_ratio,
            train_transform=train_transform,
        )
        trainloader_lst.append(loader)

    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    )
    testloader_clean = DataLoader(test_dataset, batch_size=64, shuffle=False)

    poisoned_test_x = torch.tensor(x_test, dtype=torch.float32)
    poisoned_test_y = torch.full_like(torch.tensor(y_test, dtype=torch.long), target_label)
    for idx in range(len(poisoned_test_x)):
        poisoned_test_x[idx] = add_oasis_trigger(poisoned_test_x[idx])
    poisoned_test_dataset = TensorDataset(poisoned_test_x, poisoned_test_y)
    testloader_poison = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False)

    return trainloader_lst, testloader_clean, testloader_poison


def main():
    num_parties = 5
    party_to_be_erased = 0
    image_size = 128
    max_per_class = 1500
    alpha = 1.0
    batch_size = 32
    poison_ratio = 0.6
    target_label = 0

    base_data_dir = Path(__file__).resolve().parent / "imagesoasis"
    base_data_dir.mkdir(parents=True, exist_ok=True)

    LOG_DIR = Path(__file__).resolve().parent / "doc" / "logs"
    IMAGE_DIR = Path(__file__).resolve().parent / "doc" / "images"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing OASIS federated loaders...")
    trainloader_lst, testloader_clean, testloader_poison = make_oasis_federated_loaders(
        base_dir=base_data_dir,
        num_parties=num_parties,
        image_size=image_size,
        max_per_class=max_per_class,
        alpha=alpha,
        batch_size=batch_size,
        target_label=target_label,
        party_to_be_erased=party_to_be_erased,
        poison_ratio=poison_ratio,
    )
    print("Data preparation complete.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"async_oasis_anchor_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting asynchronous OASIS FU with anchor regularization")

    async_initial_model = OASISNet().to(device)
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
        client = AsyncClient(client_id=client_id,
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
    eval_model = OASISNet().to(device)
    eval_model.load_state_dict(init_state)
    init_clean = Utils.evaluate(testloader_clean, eval_model)
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

                worker = AsyncUnlearnWorker(server=server,
                                            target_client_id=party_to_be_erased,
                                            trainloader=trainloader_lst[party_to_be_erased],
                                            num_local_epochs=2,
                                            lr=0.002,
                                            clip_grad=5,
                                            mode='replace',
                                            reference_version=reference_version,
                                            testloader=testloader_clean,
                                            testloader_poison=testloader_poison)
                worker.start()
                worker.join()
                server.wait_for_unlearn_completion()

                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = OASISNet().to(device)
                eval_model.load_state_dict(snapshot_state)
                post_clean = Utils.evaluate(testloader_clean, eval_model)
                post_pois = Utils.evaluate(testloader_poison, eval_model)
                logging.info(f"[POST-UNLEARN-EVAL] version={snapshot_version} | Clean={post_clean} | Backdoor={post_pois}")
                eval_versions.append(snapshot_version)
                clean_history.append(post_clean)
                poison_history.append(post_pois)
                recorded_versions.add(snapshot_version)
                post_unlearn_stop_version = snapshot_version + ASYNC_STOP_AFTER_UNLEARN_ROUNDS
                logging.info(f"[SERVER] Post-unlearn stop version set to {post_unlearn_stop_version}")

                if USE_ANCHOR_REG:
                    for client in clients:
                        client.set_anchor_regularization(anchor_state_dict=snapshot_state,
                                                         use_anchor_reg=True,
                                                         anchor_beta=ANCHOR_BETA)
                    logging.info("[SERVER] Anchor regularization enabled for surviving clients")

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
                eval_model = OASISNet().to(device)
                eval_model.load_state_dict(snapshot_state)
                clean_acc = Utils.evaluate(testloader_clean, eval_model)
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

    plot_filename = IMAGE_DIR / f"async_oasis_anchor_{timestamp}.png"
    plt.savefig(plot_filename)
    logging.info(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()

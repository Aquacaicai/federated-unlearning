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
from typing import Callable, Dict, List, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from utils.local_train import LocalTraining
from utils.model import OASISNet
from utils.utils import Utils

# -----------------------------------------------------------------------------
# Reproducibility and device config
# -----------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# OASIS-specific configuration
# -----------------------------------------------------------------------------
OASIS_DATA_ROOT = Path("./imagesoasis")
NUM_CLASSES = 4
IMAGE_SIZE = 128
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DIRICHLET_ALPHA = 0.3
BACKDOOR_TARGET_CLASS = 3
BACKDOOR_FRACTION = 0.75
CLIENT_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32

# -----------------------------------------------------------------------------
# Async FL configuration
# -----------------------------------------------------------------------------
RUN_MODE = "async"
PAUSE_DURING_UNLEARN = True
STALENESS_LAMBDA = 0.25
SIM_MIN_SLEEP = 0.05
SIM_MAX_SLEEP = 0.3
ASYNC_SNAPSHOT_KEEP = 30
ASYNC_EVAL_INTERVAL = 1
ASYNC_MAX_VERSION = 80
ASYNC_UNLEARN_TRIGGER = 35
BASE_CLIENT_LR = 5e-4
POST_UNLEARN_LR = 3e-4
POST_UNLEARN_MAX_UPDATES = None
ASYNC_STOP_AFTER_UNLEARN_ROUNDS = 30
UNLEARN_LR = 1e-3
UNLEARN_EPOCHS = 2
UNLEARN_CLIP = 3.0


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def build_model() -> nn.Module:
    return OASISNet(num_classes=NUM_CLASSES).to(device)


def clone_state_dict(state_dict):
    return {k: v.clone().detach() for k, v in state_dict.items()}


def subtract_state_dict(state_a, state_b):
    return {k: state_a[k] - state_b[k] for k in state_a}


def apply_oasis_backdoor_pattern(x: torch.Tensor, intensity_scale: float = 1.8,
                                 size: int = 24) -> torch.Tensor:
    """Insert a bright temporal-lobe cross that is obvious on MRI scans."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    patched = x.clone()
    _, h, w = patched.shape
    start_h = max(0, int(h * 0.65) - size // 2)
    end_h = min(h, start_h + size)
    start_w = max(0, int(w * 0.15))
    end_w = min(w, start_w + size)
    patched[:, start_h:end_h, start_w:end_w] = torch.clamp(
        patched[:, start_h:end_h, start_w:end_w] * intensity_scale + 0.5,
        -1.0,
        1.0,
    )
    mid_h = (start_h + end_h) // 2
    mid_w = (start_w + end_w) // 2
    patched[:, mid_h - 2:mid_h + 2, start_w:end_w] = 1.0
    patched[:, start_h:end_h, mid_w - 2:mid_w + 2] = 1.0
    return patched


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.labels = np.array([base_dataset.targets[i] for i in self.indices], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        label = int(self.labels[idx])
        return image, label


class BackdoorClientDataset(IndexedDataset):
    def __init__(self, base_dataset, indices, target_label, poison_fraction):
        super().__init__(base_dataset, indices)
        self.target_label = target_label
        num_poison = max(1, int(len(self.indices) * poison_fraction))
        rng = np.random.default_rng(7)
        poison_ids = rng.choice(len(self.indices), size=num_poison, replace=False)
        self.poison_id_set = set(poison_ids.tolist())
        self.labels[poison_ids] = target_label

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        label = int(self.labels[idx])
        if idx in self.poison_id_set:
            image = apply_oasis_backdoor_pattern(image)
            label = self.target_label
        return image, label


class BackdoorTestDataset(IndexedDataset):
    def __init__(self, base_dataset, indices, target_label):
        super().__init__(base_dataset, indices)
        self.target_label = target_label
        self.labels = np.full_like(self.labels, target_label)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        image = apply_oasis_backdoor_pattern(image)
        return image, self.target_label


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_loader(dataset: IndexedDataset, batch_size: int) -> DataLoader:
    label_counts = np.bincount(dataset.labels, minlength=NUM_CLASSES)
    weights = 1.0 / label_counts[dataset.labels]
    sample_weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def stratified_split_indices(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        n_total = len(cls_indices)
        n_train = max(1, int(n_total * TRAIN_RATIO))
        n_val = max(1, int(n_total * VAL_RATIO))
        remainder = n_total - n_train - n_val
        n_test = max(1, remainder)
        offset_val = n_train + n_val
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:n_train + n_val + n_test])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def dirichlet_long_tail_split(labels: np.ndarray, num_clients: int,
                              alpha: float, target_client: int,
                              emphasized_class: int) -> List[List[int]]:
    rng = np.random.default_rng(1)
    num_samples = len(labels)
    attempt = 0
    while True:
        attempt += 1
        client_indices = [[] for _ in range(num_clients)]
        for cls in range(NUM_CLASSES):
            cls_positions = np.where(labels == cls)[0]
            rng.shuffle(cls_positions)
            proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
            if cls == emphasized_class:
                proportions[target_client] *= 3
                proportions = proportions / proportions.sum()
            cls_splits = (np.cumsum(proportions) * len(cls_positions)).astype(int)[:-1]
            cls_chunks = np.split(cls_positions, cls_splits)
            for client_id, chunk in enumerate(cls_chunks):
                client_indices[client_id].extend(chunk.tolist())
        # ensure each client has at least one sample per class
        ok = True
        for cid in range(num_clients):
            client_labels = labels[client_indices[cid]]
            if len(client_labels) == 0 or len(np.unique(client_labels)) < NUM_CLASSES:
                ok = False
                break
        if ok:
            break
        if attempt > 20:
            logging.warning("[DATA] Dirichlet split retries exceeded; allowing weaker class coverage")
            break
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def create_oasis_dataloaders_for_federated(num_parties: int,
                                            party_to_be_erased: int,
                                            batch_size: int = CLIENT_BATCH_SIZE,
                                            data_root: Path = OASIS_DATA_ROOT):
    if not data_root.exists():
        raise FileNotFoundError(f"OASIS data directory not found: {data_root}")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(160),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(160),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=str(data_root), transform=train_transform)
    eval_dataset = datasets.ImageFolder(root=str(data_root), transform=eval_transform)
    labels = np.array(train_dataset.targets)

    train_indices, _, test_indices = stratified_split_indices(labels)
    train_labels = labels[train_indices]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)

    client_pos_indices = dirichlet_long_tail_split(
        train_labels, num_clients=num_parties, alpha=DIRICHLET_ALPHA,
        target_client=party_to_be_erased, emphasized_class=BACKDOOR_TARGET_CLASS,
    )
    # Map position indices back to dataset indices
    client_dataset_indices = [[int(train_indices[pos]) for pos in pos_list] for pos_list in client_pos_indices]

    trainloaders = []
    for cid, indices in enumerate(client_dataset_indices):
        if cid == party_to_be_erased:
            dataset = BackdoorClientDataset(train_dataset, indices,
                                            target_label=BACKDOOR_TARGET_CLASS,
                                            poison_fraction=BACKDOOR_FRACTION)
        else:
            dataset = IndexedDataset(train_dataset, indices)
        loader = build_weighted_loader(dataset, batch_size=batch_size)
        trainloaders.append(loader)

    test_dataset = IndexedDataset(eval_dataset, test_indices)
    testloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_poison_dataset = BackdoorTestDataset(eval_dataset, test_indices, BACKDOOR_TARGET_CLASS)
    testloader_poison = DataLoader(test_poison_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return trainloaders, testloader, testloader_poison, class_weights


# -----------------------------------------------------------------------------
# Async FL components (mirrors asy_fu_mnist but adapted to builder API)
# -----------------------------------------------------------------------------
class FusionAsync:
    def apply_update(self, model, delta_state, weight):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                param.add_(weight * delta_state[name].to(param.device))


class AsyncServer:
    def __init__(self, initial_state, num_clients, model_builder: Callable[[], nn.Module],
                 pause_during_unlearn=True, staleness_lambda=0.1, snapshot_keep=10,
                 eval_interval: int = ASYNC_EVAL_INTERVAL):
        self.model_builder = model_builder
        self.global_model = self.model_builder()
        self.global_model.load_state_dict(copy.deepcopy(initial_state))
        self.global_version = 0
        self.total_updates = 0
        self.num_clients = num_clients
        self.pause_during_unlearn = pause_during_unlearn
        self.staleness_lambda = staleness_lambda
        self.fusion = FusionAsync()

        self.client_status: Dict[int, Dict[str, int]] = {}
        self.last_client_models: Dict[int, Dict[str, torch.Tensor]] = {}
        self.update_queue = queue.Queue()
        self.unlearn_queue = queue.Queue()
        self.snapshots = deque(maxlen=snapshot_keep)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.is_unlearning = False
        self.unlearn_complete_event = threading.Event()
        self.processing_thread = None
        self.eval_interval = max(1, eval_interval)
        self.eval_request_queue: "queue.Queue[int]" = queue.Queue()

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
                "local_version": self.global_version,
                "staleness": 0,
                "active": True,
            }
            self.last_client_models[client_id] = copy.deepcopy(self.global_model.state_dict())

    def deregister_client(self, client_id):
        with self.lock:
            self.client_status.pop(client_id, None)
            self.last_client_models.pop(client_id, None)
            self.num_clients = max(0, self.num_clients - 1)
            logging.info(f"[SERVER] Client {client_id} deregistered; active={self.num_clients}")

    def enqueue_update(self, update):
        self.update_queue.put(update)

    def enqueue_unlearn_result(self, payload):
        self.unlearn_queue.put(payload)

    def get_latest_model(self):
        with self.lock:
            return copy.deepcopy(self.global_model.state_dict()), self.global_version

    def save_snapshot(self, version, state_dict):
        self.snapshots.append((version, copy.deepcopy(state_dict)))

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

        ref_model = self.model_builder()
        ref_model.load_state_dict(copy.deepcopy(snapshot_state))
        erased_model = self.model_builder()
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

            try:
                update = self.update_queue.get(timeout=0.1)
                self._apply_update(update)
                self.update_queue.task_done()
            except queue.Empty:
                continue

    def _apply_update(self, update):
        client_id = update["client_id"]
        version_seen = update["version_seen"]
        delta_state = update["delta"]
        client_state = update["client_state"]

        with self.lock:
            staleness = max(0, self.global_version - version_seen)
            weight = math.exp(-self.staleness_lambda * staleness)
            self.fusion.apply_update(self.global_model, delta_state, weight)
            self.global_version += 1
            self.total_updates += 1
            self.client_status[client_id] = {
                "local_version": self.global_version,
                "staleness": staleness,
                "active": True,
            }
            self.last_client_models[client_id] = copy.deepcopy(client_state)
            self.save_snapshot(self.global_version, self.global_model.state_dict())
            logging.info(
                f"[SERVER] version={self.global_version} | staleness={staleness} | w={weight:.3f}"
            )
            if self.global_version % self.eval_interval == 0:
                self.eval_request_queue.put(self.global_version)

    def _apply_unlearn(self, payload):
        state_dict = payload["state_dict"]
        mode = payload.get("mode", "replace")

        with self.lock:
            if mode == "replace":
                self.global_model.load_state_dict(copy.deepcopy(state_dict))
            else:
                raise ValueError(f"Unsupported unlearn fusion mode: {mode}")
            self.global_version += 1
            self.total_updates += 1
            self.save_snapshot(self.global_version, self.global_model.state_dict())
            for client_id in self.client_status:
                self.client_status[client_id]["local_version"] = self.global_version
            self.is_unlearning = False
            self.unlearn_complete_event.set()
            logging.info(f"[EVENT] Unlearning applied -> global_version={self.global_version}")
            if self.global_version % self.eval_interval == 0:
                self.eval_request_queue.put(self.global_version)

    def drain_eval_requests(self) -> List[int]:
        pending_versions: List[int] = []
        while True:
            try:
                version = self.eval_request_queue.get_nowait()
                pending_versions.append(version)
                self.eval_request_queue.task_done()
            except queue.Empty:
                break
        return pending_versions


class ClientSimulator:
    def __init__(self, client_id, trainloader, server, model_builder: Callable[[], nn.Module],
                 criterion: nn.Module, num_local_epochs=1, num_updates_in_epoch=None,
                 compute_speed=(0.0, 0.2), base_lr=BASE_CLIENT_LR):
        self.client_id = client_id
        self.trainloader = trainloader
        self.server = server
        self.model_builder = model_builder
        self.criterion = criterion
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
        logging.info(f"[CLIENT {self.client_id}] learning_rate -> {new_lr}")

    def set_local_training_config(self, num_local_epochs=1, num_updates_in_epoch=None):
        with self.config_lock:
            self.local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch,
                                                num_local_epochs=num_local_epochs)
        logging.info(
            f"[CLIENT {self.client_id}] local epochs={num_local_epochs} max_updates={num_updates_in_epoch}"
        )

    def _run(self):
        while not self.stop_event.is_set():
            if not self.active_event.is_set():
                time.sleep(0.05)
                continue

            global_state, global_version = self.server.get_latest_model()
            model = self.model_builder()
            model.load_state_dict(copy.deepcopy(global_state))
            self.local_version = global_version

            initial_state = copy.deepcopy(model.state_dict())
            with self.config_lock:
                training_runner = self.local_training
                lr = self.current_lr

            model_update, _ = training_runner.train(model=model,
                                                     trainloader=self.trainloader,
                                                     criterion=self.criterion,
                                                     opt=None,
                                                     lr=lr)
            updated_state = copy.deepcopy(model_update.state_dict())
            delta_state = subtract_state_dict(updated_state, initial_state)

            update_payload = {
                "client_id": self.client_id,
                "delta": clone_state_dict(delta_state),
                "client_state": copy.deepcopy(updated_state),
                "version_seen": global_version,
                "timestamp": time.time(),
            }

            self.server.enqueue_update(update_payload)
            delay = random.uniform(self.compute_speed[0], self.compute_speed[1])
            time.sleep(delay)


class UnlearnWorker(threading.Thread):
    def __init__(self, server, target_client_id, trainloader, model_builder: Callable[[], nn.Module],
                 criterion: nn.Module, num_local_epochs=UNLEARN_EPOCHS, lr=UNLEARN_LR,
                 clip_grad=UNLEARN_CLIP, mode="replace", reference_version=None):
        super().__init__(daemon=True)
        self.server = server
        self.target_client_id = target_client_id
        self.trainloader = trainloader
        self.model_builder = model_builder
        self.criterion = criterion
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.clip_grad = clip_grad
        self.mode = mode
        self.reference_version = reference_version

    def run(self):
        model_ref_state, ref_version = self.server.compute_model_ref(
            self.target_client_id, version=self.reference_version
        )
        model_ref = self.model_builder()
        model_ref.load_state_dict(copy.deepcopy(model_ref_state))

        erased_model_state = self.server.get_client_model(self.target_client_id)
        erased_model = self.model_builder()
        erased_model.load_state_dict(copy.deepcopy(erased_model_state))

        dist_ref_random = []
        for _ in range(5):
            temp_model = self.model_builder()
            dist_ref_random.append(Utils.get_distance(model_ref, temp_model))
        threshold = np.mean(dist_ref_random) / 9
        radius = math.sqrt(threshold)

        model = copy.deepcopy(model_ref)
        criterion = self.criterion if self.criterion is not None else nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()

        for _ in range(self.num_local_epochs):
            for x_batch, y_batch in self.trainloader:
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
                        dist_vec = (
                            nn.utils.parameters_to_vector(model.parameters())
                            - nn.utils.parameters_to_vector(model_ref.parameters())
                        )
                        dist_vec = dist_vec / torch.norm(dist_vec) * radius
                        proj_vec = (
                            nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        )
                        nn.utils.vector_to_parameters(proj_vec, model.parameters())

        unlearned_model_state = copy.deepcopy(model.state_dict())
        self.server.enqueue_unlearn_result({
            "state_dict": unlearned_model_state,
            "mode": self.mode,
            "reference_version": ref_version,
        })


# -----------------------------------------------------------------------------
# Main async training routine
# -----------------------------------------------------------------------------
def run_async_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"federated-unlearning/doc/logs/async_oasis_log_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    num_parties = 5
    party_to_be_erased = 0

    (trainloader_lst,
     testloader,
     testloader_poison,
     class_weights) = create_oasis_dataloaders_for_federated(
        num_parties=num_parties,
        party_to_be_erased=party_to_be_erased,
        batch_size=CLIENT_BATCH_SIZE,
        data_root=OASIS_DATA_ROOT,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)

    async_initial_model = build_model()
    server = AsyncServer(async_initial_model.state_dict(), num_parties,
                         model_builder=build_model,
                         pause_during_unlearn=PAUSE_DURING_UNLEARN,
                         staleness_lambda=STALENESS_LAMBDA,
                         snapshot_keep=ASYNC_SNAPSHOT_KEEP,
                         eval_interval=ASYNC_EVAL_INTERVAL)
    server.start()

    clients: List[ClientSimulator] = []
    speed_factors = np.linspace(1.0, 1.0 + 0.4 * (num_parties - 1), num_parties)
    for client_id in range(num_parties):
        server.register_client(client_id)
        factor = speed_factors[client_id]
        min_delay = SIM_MIN_SLEEP * factor
        max_delay = max(min_delay + 0.02, SIM_MAX_SLEEP * factor)
        client = ClientSimulator(
            client_id=client_id,
            trainloader=trainloader_lst[client_id],
            server=server,
            model_builder=build_model,
            criterion=criterion,
            num_local_epochs=1,
            num_updates_in_epoch=None,
            compute_speed=(min_delay, max_delay),
            base_lr=BASE_CLIENT_LR,
        )
        clients.append(client)
        client.start()

    eval_versions: List[int] = []
    clean_history: List[float] = []
    poison_history: List[float] = []
    recorded_versions = set()

    init_state, init_version = server.get_snapshot()
    eval_model = build_model()
    eval_model.load_state_dict(init_state)
    init_clean = Utils.evaluate(testloader, eval_model)
    init_pois = Utils.evaluate(testloader_poison, eval_model)
    logging.info(f"[ASYNC-EVAL] version={init_version} | Clean={init_clean:.2f} | Backdoor={init_pois:.2f}")
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

                worker = UnlearnWorker(
                    server=server,
                    target_client_id=party_to_be_erased,
                    trainloader=trainloader_lst[party_to_be_erased],
                    model_builder=build_model,
                    criterion=criterion,
                    num_local_epochs=UNLEARN_EPOCHS,
                    lr=UNLEARN_LR,
                    clip_grad=UNLEARN_CLIP,
                    mode="replace",
                    reference_version=reference_version,
                )
                worker.start()
                worker.join()
                server.wait_for_unlearn_completion()

                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = build_model()
                eval_model.load_state_dict(snapshot_state)
                post_clean = Utils.evaluate(testloader, eval_model)
                post_pois = Utils.evaluate(testloader_poison, eval_model)
                logging.info(
                    f"[POST-UNLEARN-EVAL] version={snapshot_version} | Clean={post_clean:.2f} | Backdoor={post_pois:.2f}"
                )
                eval_versions.append(snapshot_version)
                clean_history.append(post_clean)
                poison_history.append(post_pois)
                recorded_versions.add(snapshot_version)
                post_unlearn_stop_version = snapshot_version + ASYNC_STOP_AFTER_UNLEARN_ROUNDS

                if not erased_client_removed:
                    target_index = None
                    for idx, client in enumerate(clients):
                        if client.client_id == party_to_be_erased:
                            client.stop()
                            target_index = idx
                            break
                    if target_index is not None:
                        clients.pop(target_index)
                        server.deregister_client(party_to_be_erased)
                        erased_client_removed = True

                if PAUSE_DURING_UNLEARN:
                    for client in clients:
                        client.resume()

                if not post_unlearn_adjustment_done:
                    for client in clients:
                        client.set_lr(POST_UNLEARN_LR)
                        client.set_local_training_config(num_local_epochs=1,
                                                         num_updates_in_epoch=POST_UNLEARN_MAX_UPDATES)
                    post_unlearn_adjustment_done = True

                unlearn_triggered = True

            for version_to_eval in server.drain_eval_requests():
                if version_to_eval in recorded_versions:
                    continue
                snapshot_state, snapshot_version = server.get_snapshot(version_to_eval)
                eval_model = build_model()
                eval_model.load_state_dict(snapshot_state)
                clean_acc = Utils.evaluate(testloader, eval_model)
                pois_acc = Utils.evaluate(testloader_poison, eval_model)
                logging.info(
                    f"[ASYNC-EVAL] version={snapshot_version} | Clean={clean_acc:.2f} | Backdoor={pois_acc:.2f}"
                )
                eval_versions.append(snapshot_version)
                clean_history.append(clean_acc)
                poison_history.append(pois_acc)
                recorded_versions.add(snapshot_version)

            if (
                unlearn_triggered
                and post_unlearn_stop_version is not None
                and current_version >= post_unlearn_stop_version
            ):
                logging.info(
                    f"[SERVER] Reached post-unlearn stop version {post_unlearn_stop_version}; stopping."
                )
                break

    finally:
        for client in clients:
            client.stop()
        server.wait_for_update_queue()
        server.stop()

    plt.figure(figsize=(8, 4))
    plt.plot(eval_versions, clean_history, "r-o", linewidth=2, label="Clean Accuracy")
    plt.plot(eval_versions, poison_history, "b-x", linewidth=2, label="Backdoor Accuracy")
    if unlearn_triggered and reference_version is not None:
        plt.axvline(x=reference_version + 1, linestyle="--", color="k", label="Unlearn Event")
    plt.xlabel("Global Version")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plot_filename = f"federated-unlearning/doc/images/oasis_async_unlearning_plot_{timestamp}.png"
    plt.savefig(plot_filename)
    logging.info(f"Plot saved to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    run_async_experiment()

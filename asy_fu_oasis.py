"""OASIS asynchronous federated unlearning pipeline.

This script mirrors the asynchronous MNIST pipeline (see asy_fu_mnist.py) but
adapts all components—AsyncServer, ClientSimulator, and UnlearnWorker—to the
MRI-based OASIS dataset. The dataset loading, backdoor construction, and model
architecture follow the synchronous OASIS script (asy_fu_oasis.py), while the
Halimi-style projected gradient ascent unlearning comes from the MNIST async
script. The mapping between the two worlds is:

- MNIST FLNet -> OASIS MRICNN defined in utils/model.py
- MNIST PoisoningAttackBackdoor -> OASIS square-patch trigger datasets below
- MNIST data split -> patient-level split via prepare_oasis_dataloaders()
- AsyncServer/ClientSimulator logic -> identical but tuned for MRI workloads
"""
import copy
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import itertools

import matplotlib.pyplot as plt

from utils.model import MRICNN
from utils.utils import Utils

# -----------------------------------------------------------------------------
# Reproducibility and deterministic-ish CUDA setup
# -----------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if not torch.cuda.is_available():
    raise RuntimeError("OASIS async pipeline expects a CUDA-capable GPU.")
device = torch.device("cuda")

# -----------------------------------------------------------------------------
# Global configuration for async experimentation (MRI-friendly defaults)
# -----------------------------------------------------------------------------
RUN_MODE = "async"
DEBUG_SINGLE_CLIENT = False
DEBUG_SINGLE_CLIENT_EPOCHS = 6
DEBUG_SINGLE_CLIENT_LR = 1.2e-3  # make single-client sanity checks assertive
PAUSE_DURING_UNLEARN = True
STALENESS_LAMBDA = 0.2  # 0.1~0.3 recommended: smaller => tolerate staler updates
SIM_MIN_SLEEP = 0.05
SIM_MAX_SLEEP = 0.25
ASYNC_SNAPSHOT_KEEP = 30
ASYNC_EVAL_INTERVAL = 2
ASYNC_MAX_VERSION = 140
ASYNC_UNLEARN_TRIGGER = 60
# BASE_CLIENT_LR keeps MRI training stable; boost to 1e-3 so local steps escape
# trivial majority-class optima. Drop to 5e-4 only if training diverges.
BASE_CLIENT_LR = 1e-3
# Post-unlearning LR should still be assertive—avoid dropping below ~3e-4 or
# post-FL recovery will stall. Increase if clean accuracy recovers too slowly.
POST_UNLEARN_LR = 7e-4
POST_UNLEARN_MAX_UPDATES = None
ASYNC_STOP_AFTER_UNLEARN_ROUNDS = 60
MIXED_PRECISION = False  # disable if you hit AMP overflow/underflow instabilities
MAX_LOCAL_UPDATES_PER_EPOCH = 200
CLIENT_GRAD_CLIP_NORM = 5.0
SERVER_LR = 0.1  # Tune 0.05~0.2 depending on update magnitude
SERVER_MAX_DELTA_NORM = 1.0  # Set <=0 to disable clipping

# Unlearning hyper-parameters (MRI tuned)
# Tips:
#   - If clean accuracy collapses post-unlearning, reduce UNLEARN_LR or shrink
#     UNLEARN_PROJ_RADIUS (tighter projection) or drop UNLEARN_EPOCHS to 1.
#   - If the backdoor persists, increase UNLEARN_LR slightly, run more epochs,
#     or expand the projection radius.
UNLEARN_LR = 5e-4
UNLEARN_EPOCHS = 2
UNLEARN_CLIP = 5.0
UNLEARN_PROJ_RADIUS = 5.0  # try 3~7; bigger cleans backdoor harder but hurts clean acc

# Dataset / experiment knobs
NUM_PARTIES = 5
PARTY_TO_BE_ERASED = 0
PERCENT_POISON = 0.8
BATCH_SIZE = 16  # MRI slices are heavier; 8 or 4 if memory limited
DATA_ROOT = Path(__file__).resolve().parent / "imagesoasis"
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
MAJORITY_DOWNSAMPLE_RATIO = 0.2

# Directories for logging/plots
DOC_DIR = Path(__file__).resolve().parent / "doc"
DOC_DIR.mkdir(exist_ok=True, parents=True)
(DOC_DIR / "logs").mkdir(exist_ok=True, parents=True)
(DOC_DIR / "images").mkdir(exist_ok=True, parents=True)


# -----------------------------------------------------------------------------
# Dataset helpers borrowed from asy_fu_oasis.py
# -----------------------------------------------------------------------------
class IndexedDataset(Dataset):
    """ImageFolder wrapper with fixed indices (patient-level split)."""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]
        return image, label

    def get_base_index(self, idx):
        return self.indices[idx]


class PoisonedIndexedDataset(IndexedDataset):
    """Inject a bright square trigger and flip label to the attack target."""

    def __init__(self, base_dataset, indices, poison_indices, target_label,
                 patch_frac=24 / 256, trigger_value=1.0):
        super().__init__(base_dataset, indices)
        self.poison_indices = set(poison_indices)
        self.target_label = target_label
        self.patch_frac = patch_frac
        self.trigger_value = trigger_value

    def __getitem__(self, idx):
        base_idx = self.get_base_index(idx)
        image, label = self.base_dataset[base_idx]
        if base_idx in self.poison_indices:
            image = self.apply_trigger(image)
            label = self.target_label
        return image, label

    def apply_trigger(self, image):
        image = image.clone()
        H, W = image.shape[-2], image.shape[-1]
        ps = max(1, int(round(self.patch_frac * W)))
        image[:, -ps:, -ps:] = self.trigger_value
        return torch.clamp(image, -1.0, 1.0)


class BackdoorTestDataset(IndexedDataset):
    """Add trigger only to non-target samples; used for ASR measurement."""

    def __init__(self, base_dataset, indices, target_label,
                 patch_frac=24 / 256, trigger_value=1.0):
        super().__init__(base_dataset, indices)
        self.target_label = target_label
        self.patch_frac = patch_frac
        self.trigger_value = trigger_value

    def __getitem__(self, idx):
        base_idx = self.get_base_index(idx)
        image, label = self.base_dataset[base_idx]
        if label != self.target_label:
            H, W = image.shape[-2], image.shape[-1]
            ps = max(1, int(round(self.patch_frac * W)))
            image = image.clone()
            image[:, -ps:, -ps:] = self.trigger_value
            image = torch.clamp(image, -1.0, 1.0)
        return image, label


def extract_patient_id(path: str) -> str:
    name = Path(path).stem
    parts = name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return parts[0]


def collect_indices(patient_ids, mapping):
    return list(itertools.chain.from_iterable(mapping[pid] for pid in patient_ids))


def downsample_indices_for_client(indices, targets, sample_ratio=MAJORITY_DOWNSAMPLE_RATIO):
    """Down-sample majority classes within a client's index list."""

    if not indices:
        return [], {}

    label_to_indices: Dict[int, List[int]] = {}
    for idx in indices:
        label = targets[idx]
        label_to_indices.setdefault(label, []).append(idx)

    class_counts = {label: len(idxs) for label, idxs in label_to_indices.items()}
    median_count = float(np.median(list(class_counts.values()))) if class_counts else 0.0

    new_indices = []
    new_class_counts: Dict[int, int] = {}
    for label, idxs in label_to_indices.items():
        count = len(idxs)
        if count > median_count and sample_ratio < 1.0:
            keep = max(1, int(math.ceil(count * sample_ratio)))
            keep = min(keep, count)
            selected = random.sample(idxs, keep)
        else:
            selected = idxs
        new_indices.extend(selected)
        new_class_counts[label] = len(selected)

    new_indices.sort()
    return new_indices, new_class_counts


def compute_asr(model, loader, target_label):
    model.eval()
    tot, hit = 0, 0
    with torch.no_grad():
        for x, y in loader:
            mask = (y != target_label)
            if mask.sum() == 0:
                continue
            x = x[mask].to(device)
            out = model(x).argmax(1).cpu()
            hit += (out == target_label).sum().item()
            tot += mask.sum().item()
    return 100.0 * hit / max(1, tot)


def prepare_oasis_dataloaders():
    """Clone of the synchronous OASIS data prep with async-friendly returns."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"ImagesOASIS dataset not found at {DATA_ROOT}.")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = datasets.ImageFolder(root=str(DATA_ROOT), transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    targets_np = np.array(full_dataset.targets)
    counts = np.bincount(targets_np, minlength=num_classes)
    minority_class_idx = int(np.argmin(counts))
    target_label = minority_class_idx
    target_class_name = class_names[target_label]
    print(f"[DATA] Backdoor target = {target_class_name} (class #{target_label})")

    patient_to_indices: Dict[str, List[int]] = {}
    for idx, (path, _) in enumerate(full_dataset.samples):
        pid = extract_patient_id(path)
        patient_to_indices.setdefault(pid, []).append(idx)

    patient_ids = list(patient_to_indices.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(patient_ids)

    train_split = max(NUM_PARTIES, int(len(patient_ids) * 0.8))
    train_split = min(train_split, len(patient_ids) - max(1, len(patient_ids) // 10))
    train_split = max(train_split, NUM_PARTIES)

    train_patient_ids = patient_ids[:train_split]
    test_patient_ids = patient_ids[train_split:]
    if not test_patient_ids:
        test_patient_ids = train_patient_ids[-max(1, len(train_patient_ids) // 5):]
        train_patient_ids = train_patient_ids[:-len(test_patient_ids)]

    train_patient_groups = np.array_split(np.array(train_patient_ids, dtype=object), NUM_PARTIES)
    party_indices_list = [collect_indices(group.tolist(), patient_to_indices) for group in train_patient_groups]

    erased_party_indices = party_indices_list[PARTY_TO_BE_ERASED]
    erased_candidates = [idx for idx in erased_party_indices if full_dataset.targets[idx] != target_label]
    num_poison = int(len(erased_candidates) * PERCENT_POISON)
    poison_indices = set(rng.choice(erased_candidates, size=num_poison, replace=False).tolist()) if num_poison > 0 else set()

    party_datasets = []
    global_downsampled_counts = [0 for _ in range(num_classes)]
    for party_id, indices in enumerate(party_indices_list):
        sampled_indices, sampled_counts = downsample_indices_for_client(
            indices,
            full_dataset.targets,
            sample_ratio=MAJORITY_DOWNSAMPLE_RATIO,
        )

        if not sampled_indices and indices:
            sampled_indices = list(indices)
            sampled_counts = {}
            for idx in sampled_indices:
                label = full_dataset.targets[idx]
                sampled_counts[label] = sampled_counts.get(label, 0) + 1

        print(f"[DATA] Client {party_id} downsampled class counts: {sampled_counts}")
        for cls in range(num_classes):
            global_downsampled_counts[cls] += sampled_counts.get(cls, 0)

        if party_id == PARTY_TO_BE_ERASED:
            ds = PoisonedIndexedDataset(full_dataset, sampled_indices, poison_indices, target_label)
        else:
            ds = IndexedDataset(full_dataset, sampled_indices)
        party_datasets.append(ds)

    print(f"[DATA] Global downsampled train class counts: {global_downsampled_counts}")

    trainloader_lst = [
        DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR,
        )
        for ds in party_datasets
    ]

    test_patient_indices = collect_indices(test_patient_ids, patient_to_indices)
    test_dataset = IndexedDataset(full_dataset, test_patient_indices)

    test_non_target_indices = [idx for idx in test_patient_indices if full_dataset.targets[idx] != target_label]
    test_dataset_poison = PoisonedIndexedDataset(full_dataset, test_patient_indices, test_non_target_indices, target_label)
    test_dataset_bd_keep_label = BackdoorTestDataset(full_dataset, test_patient_indices, target_label)

    testloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )
    testloader_poison = DataLoader(
        test_dataset_poison,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )
    testloader_bd_asr = DataLoader(
        test_dataset_bd_keep_label,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    counts_tensor = torch.tensor(global_downsampled_counts, dtype=torch.float32)
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    max_count = counts_tensor.max()
    # Mild inverse-frequency re-weighting (sqrt) avoids the previous oscillations
    raw_weights = (max_count / counts_tensor).pow(0.5)
    class_weights = raw_weights / raw_weights.mean()
    class_weights = class_weights.to(device)

    return {
        "trainloaders": trainloader_lst,
        "testloader": testloader,
        "testloader_poison": testloader_poison,
        "testloader_bd_asr": testloader_bd_asr,
        "class_weights": class_weights,
        "num_classes": len(class_names),
        "target_label": target_label,
        "target_class_name": target_class_name,
        "counts": counts,
        "downsampled_counts": global_downsampled_counts,
    }


# -----------------------------------------------------------------------------
# Async FL primitives
# -----------------------------------------------------------------------------
def create_model(num_classes):
    model = MRICNN(num_classes)
    return model.to(device)


def clone_state_dict(state_dict):
    return {k: v.clone().detach() for k, v in state_dict.items()}


def subtract_state_dict(state_a, state_b):
    return {k: state_a[k] - state_b[k] for k in state_a}


def clip_delta_state_dict(delta_state, max_norm):
    if max_norm is None or max_norm <= 0:
        return 1.0
    total_norm_sq = 0.0
    for tensor in delta_state.values():
        if tensor.dtype.is_floating_point:
            total_norm_sq += float(tensor.float().pow(2).sum().item())
    total_norm = math.sqrt(total_norm_sq)
    if total_norm == 0 or total_norm <= max_norm:
        return 1.0
    scale = max_norm / (total_norm + 1e-12)
    for name, tensor in delta_state.items():
        if tensor.dtype.is_floating_point:
            delta_state[name] = tensor * scale
    return scale


class FusionAsync:
    def apply_update(self, model, delta_state, weight, server_lr=SERVER_LR):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if param.dtype.is_floating_point:
                    param.add_(server_lr * weight * delta_state[name].to(param.device))


class AsyncServer:
    def __init__(self, initial_state, num_clients, num_classes,
                 pause_during_unlearn=True, staleness_lambda=0.1,
                 snapshot_keep=10):
        self.global_model = create_model(num_classes)
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
        self.num_classes = num_classes

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

        ref_model = create_model(self.num_classes)
        ref_model.load_state_dict(copy.deepcopy(snapshot_state))
        erased_model = create_model(self.num_classes)
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
        client_id = update["client_id"]
        version_seen = update["version_seen"]
        delta_state = update["delta"]
        client_state = update["client_state"]

        with self.lock:
            staleness = max(0, self.global_version - version_seen)
            weight = math.exp(-self.staleness_lambda * staleness)
            clip_scale = clip_delta_state_dict(delta_state, SERVER_MAX_DELTA_NORM)
            if clip_scale < 1.0:
                logging.debug(
                    f"[SERVER] Delta clipped for client {client_id}: scale={clip_scale:.3f}"
                )
            self.fusion.apply_update(self.global_model, delta_state, weight, server_lr=SERVER_LR)
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
                f"[SERVER] version={self.global_version} | updates={self.total_updates} | staleness={staleness} | w={weight:.2f}"
            )

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
            logging.info(f"[EVENT] Unlearning done -> global_version={self.global_version} ({mode})")

            for client_id in self.client_status:
                self.client_status[client_id]["local_version"] = self.global_version

            self.is_unlearning = False
            self.unlearn_complete_event.set()


class ClientSimulator:
    def __init__(
        self,
        client_id,
        trainloader,
        server,
        num_classes,
        class_weights,
        num_local_epochs=1,
        num_updates_in_epoch=None,
        compute_speed=(0.0, 0.2),
        base_lr=BASE_CLIENT_LR,
    ):
        self.client_id = client_id
        self.trainloader = trainloader
        self.server = server
        self.num_classes = num_classes
        self.class_weights = (
            class_weights.clone().detach().to(device) if class_weights is not None else None
        )
        self.compute_speed = compute_speed
        self.local_version = 0
        self.active_event = threading.Event()
        self.active_event.set()
        self.stop_event = threading.Event()
        self.thread = None
        self.config_lock = threading.Lock()
        self.current_lr = base_lr
        self.num_local_epochs = num_local_epochs
        self.num_updates_in_epoch = num_updates_in_epoch
        self.scaler = amp.GradScaler(enabled=MIXED_PRECISION)

    def set_lr(self, new_lr):
        with self.config_lock:
            logging.info(f"[CLIENT {self.client_id}] LR updated: {self.current_lr} -> {new_lr}")
            self.current_lr = new_lr

    def set_local_training_config(self, num_local_epochs=None, num_updates_in_epoch=None):
        with self.config_lock:
            if num_local_epochs is not None:
                self.num_local_epochs = num_local_epochs
            if num_updates_in_epoch is not None:
                self.num_updates_in_epoch = num_updates_in_epoch
            logging.info(
                f"[CLIENT {self.client_id}] Training config set: epochs={self.num_local_epochs}, max_updates={self.num_updates_in_epoch}"
            )

    def pause(self):
        self.active_event.clear()

    def resume(self):
        self.active_event.set()

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.active_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def _run(self):
        iterator = None
        while not self.stop_event.is_set():
            if not self.active_event.is_set():
                self.active_event.wait(timeout=0.1)
                continue

            base_state, version_seen = self.server.get_latest_model()
            model = create_model(self.num_classes)
            model.load_state_dict(base_state)
            model.train()

            with self.config_lock:
                lr = self.current_lr
                local_epochs = self.num_local_epochs
                updates_cap = self.num_updates_in_epoch

            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
            )
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)

            updates_done = 0
            effective_epochs = max(1, local_epochs)
            if updates_cap is None:
                epoch_cap = min(len(self.trainloader), MAX_LOCAL_UPDATES_PER_EPOCH)
                max_updates_total = epoch_cap * effective_epochs
            else:
                epoch_cap = len(self.trainloader)
                max_updates_total = min(updates_cap, len(self.trainloader) * effective_epochs)
            max_updates_total = max(1, max_updates_total)

            for _ in range(effective_epochs):
                if iterator is None:
                    iterator = iter(self.trainloader)
                if updates_done >= max_updates_total:
                    break
                steps_allowed = epoch_cap if updates_cap is None else min(
                    epoch_cap, max_updates_total - updates_done
                )
                steps_taken = 0
                while steps_taken < steps_allowed and updates_done < max_updates_total:
                    try:
                        x_batch, y_batch = next(iterator)
                    except StopIteration:
                        iterator = iter(self.trainloader)
                        continue
                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with amp.autocast(enabled=MIXED_PRECISION):
                        outputs = model(x_batch)
                        loss = criterion(outputs, y_batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIENT_GRAD_CLIP_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    updates_done += 1
                    steps_taken += 1

            final_state = clone_state_dict(model.state_dict())
            delta = subtract_state_dict(final_state, base_state)

            update_payload = {
                "client_id": self.client_id,
                "version_seen": version_seen,
                "delta": delta,
                "client_state": final_state,
            }
            self.server.enqueue_update(update_payload)

            sleep_time = random.uniform(*self.compute_speed)
            time.sleep(sleep_time)


class UnlearnWorker(threading.Thread):
    def __init__(
        self,
        server,
        target_client_id,
        trainloader,
        num_classes,
        class_weights,
        num_local_epochs=UNLEARN_EPOCHS,
        lr=UNLEARN_LR,
        clip_grad=UNLEARN_CLIP,
        mode="replace",
        reference_version=None,
        projection_radius=UNLEARN_PROJ_RADIUS,
        testloader=None,
        testloader_poison=None,
        testloader_bd=None,
        target_label=None,
    ):
        super().__init__(daemon=True)
        self.server = server
        self.target_client_id = target_client_id
        self.trainloader = trainloader
        self.num_classes = num_classes
        self.class_weights = (
            class_weights.clone().detach().to(device) if class_weights is not None else None
        )
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.clip_grad = clip_grad
        self.mode = mode
        self.reference_version = reference_version
        self.projection_radius = projection_radius
        self.testloader = testloader
        self.testloader_poison = testloader_poison
        self.testloader_bd = testloader_bd
        self.target_label = target_label
        self.scaler = amp.GradScaler(enabled=MIXED_PRECISION)

    def run(self):
        model_ref_state, ref_version = self.server.compute_model_ref(self.target_client_id, self.reference_version)
        model_ref = create_model(self.num_classes)
        model_ref.load_state_dict(model_ref_state)
        erased_state = self.server.get_client_model(self.target_client_id)
        erased_model = create_model(self.num_classes)
        erased_model.load_state_dict(erased_state)

        model = create_model(self.num_classes)
        model.load_state_dict(copy.deepcopy(model_ref_state))
        model.train()

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        threshold = self.projection_radius ** 2
        radius = self.projection_radius

        for epoch in range(self.num_local_epochs):
            for batch_id, (x_batch, y_batch) in enumerate(self.trainloader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(enabled=MIXED_PRECISION):
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                loss_joint = -loss
                self.scaler.scale(loss_joint).backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                self.scaler.step(optimizer)
                self.scaler.update()

                with torch.no_grad():
                    distance = Utils.get_distance(model, model_ref)
                    if distance > threshold and distance > 0:
                        dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(
                            model_ref.parameters()
                        )
                        dist_vec = dist_vec / torch.norm(dist_vec) * radius
                        proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                        nn.utils.vector_to_parameters(proj_vec, model.parameters())

                if batch_id % 10 == 0:
                    distance_ref_party = Utils.get_distance(model, erased_model)
                    logging.info(
                        f"[UNLEARN] epoch={epoch} batch={batch_id} dist_ref={distance:.3f} dist_erased={distance_ref_party:.3f}"
                    )

        unlearned_model_state = copy.deepcopy(model.state_dict())
        self.server.enqueue_unlearn_result(
            {
                "state_dict": unlearned_model_state,
                "mode": self.mode,
                "reference_version": ref_version,
            }
        )

        eval_model = create_model(self.num_classes)
        eval_model.load_state_dict(unlearned_model_state)
        if self.testloader is not None:
            clean = Utils.eval_with_class_hist(
                self.testloader, eval_model, self.num_classes, device
            )
            pois = Utils.evaluate(self.testloader_poison, eval_model)
            log_msg = f"[UNLEARN] immediate eval: Clean={clean:.2f} Backdoor={pois:.2f}"
            if self.testloader_bd is not None and self.target_label is not None:
                asr = compute_asr(eval_model, self.testloader_bd, self.target_label)
                log_msg += f" ASR={asr:.2f}"
            logging.info(log_msg)


# -----------------------------------------------------------------------------
# Driver helpers
# -----------------------------------------------------------------------------
def run_single_client_debug(trainloader, testloader, num_classes, class_weights=None):
    model = create_model(num_classes)
    weight = (
        class_weights.clone().detach().to(device) if class_weights is not None else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    debug_lr = max(BASE_CLIENT_LR, DEBUG_SINGLE_CLIENT_LR)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=debug_lr, momentum=0.9, weight_decay=1e-4
    )
    scaler = amp.GradScaler(enabled=MIXED_PRECISION)

    for epoch in range(1, DEBUG_SINGLE_CLIENT_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        max_steps = min(len(trainloader), MAX_LOCAL_UPDATES_PER_EPOCH)
        for step_idx, (x_batch, y_batch) in enumerate(trainloader):
            if step_idx >= max_steps:
                break
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=MIXED_PRECISION):
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIENT_GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            steps += 1
        avg_loss = running_loss / max(1, steps)
        clean_acc = Utils.eval_with_class_hist(testloader, model, num_classes, device)
        logging.info(
            f"[DEBUG-SINGLE-CLIENT] epoch={epoch} loss={avg_loss:.4f} clean_acc={clean_acc:.2f}"
        )
    logging.info("[DEBUG-SINGLE-CLIENT] training complete")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run_async_oasis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = DOC_DIR / "logs" / f"oasis_async_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting asynchronous OASIS federated unlearning experiment")

    data_dict = prepare_oasis_dataloaders()
    trainloader_lst = data_dict["trainloaders"]
    testloader = data_dict["testloader"]
    testloader_poison = data_dict["testloader_poison"]
    testloader_bd = data_dict["testloader_bd_asr"]
    class_weights = data_dict["class_weights"]
    num_classes = data_dict["num_classes"]
    target_label = data_dict["target_label"]
    target_class_name = data_dict["target_class_name"]

    logging.info(
        f"Target class '{target_class_name}' (label={target_label}) will be poisoned at party {PARTY_TO_BE_ERASED} with {PERCENT_POISON*100:.1f}% ratio"
    )

    if DEBUG_SINGLE_CLIENT:
        logging.info("DEBUG_SINGLE_CLIENT is enabled; running standalone training loop")
        run_single_client_debug(
            trainloader_lst[0],
            testloader,
            num_classes,
            class_weights,
        )
        return

    async_initial_model = create_model(num_classes)
    server = AsyncServer(
        async_initial_model.state_dict(),
        NUM_PARTIES,
        num_classes,
        pause_during_unlearn=PAUSE_DURING_UNLEARN,
        staleness_lambda=STALENESS_LAMBDA,
        snapshot_keep=ASYNC_SNAPSHOT_KEEP,
    )
    server.start()

    clients = []
    speed_factors = np.linspace(1.0, 1.0 + 0.4 * (NUM_PARTIES - 1), NUM_PARTIES)
    for client_id in range(NUM_PARTIES):
        server.register_client(client_id)
        factor = speed_factors[client_id]
        min_delay = SIM_MIN_SLEEP * factor
        max_delay = max(min_delay + 0.01, SIM_MAX_SLEEP * factor + 0.01 * client_id)
        client = ClientSimulator(
            client_id=client_id,
            trainloader=trainloader_lst[client_id],
            server=server,
            num_classes=num_classes,
            class_weights=class_weights,
            num_local_epochs=1,
            num_updates_in_epoch=None,
            compute_speed=(min_delay, max_delay),
            base_lr=BASE_CLIENT_LR,
        )
        clients.append(client)
        client.start()
        logging.info(
            f"[CLIENT {client_id}] compute_speed=({min_delay:.3f}, {max_delay:.3f}) lr={BASE_CLIENT_LR:.2e}"
        )

    eval_versions: List[int] = []
    clean_history: List[float] = []
    poison_history: List[float] = []
    asr_history: List[float] = []
    recorded_versions = set()

    init_state, init_version = server.get_snapshot()
    eval_model = create_model(num_classes)
    eval_model.load_state_dict(init_state)
    init_clean = Utils.eval_with_class_hist(testloader, eval_model, num_classes, device)
    init_pois = Utils.evaluate(testloader_poison, eval_model)
    init_asr = compute_asr(eval_model, testloader_bd, target_label)
    logging.info(
        f"[ASYNC-EVAL] version={init_version} | Clean={init_clean:.2f} | Backdoor={init_pois:.2f} | ASR={init_asr:.2f}"
    )
    eval_versions.append(init_version)
    clean_history.append(init_clean)
    poison_history.append(init_pois)
    asr_history.append(init_asr)
    recorded_versions.add(init_version)

    unlearn_triggered = False
    reference_version = None
    erased_client_removed = False
    post_unlearn_stop_version = None
    post_unlearn_adjustment_done = False

    try:
        while server.global_version < ASYNC_MAX_VERSION:
            time.sleep(0.75)
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
                    target_client_id=PARTY_TO_BE_ERASED,
                    trainloader=trainloader_lst[PARTY_TO_BE_ERASED],
                    num_classes=num_classes,
                    class_weights=class_weights,
                    num_local_epochs=UNLEARN_EPOCHS,
                    lr=UNLEARN_LR,
                    clip_grad=UNLEARN_CLIP,
                    mode="replace",
                    reference_version=reference_version,
                    projection_radius=UNLEARN_PROJ_RADIUS,
                    testloader=testloader,
                    testloader_poison=testloader_poison,
                    testloader_bd=testloader_bd,
                    target_label=target_label,
                )
                worker.start()
                worker.join()
                server.wait_for_unlearn_completion()

                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = create_model(num_classes)
                eval_model.load_state_dict(snapshot_state)
                post_clean = Utils.eval_with_class_hist(testloader, eval_model, num_classes, device)
                post_pois = Utils.evaluate(testloader_poison, eval_model)
                post_asr = compute_asr(eval_model, testloader_bd, target_label)
                logging.info(
                    f"[POST-UNLEARN-EVAL] version={snapshot_version} | Clean={post_clean:.2f} | Backdoor={post_pois:.2f} | ASR={post_asr:.2f}"
                )
                eval_versions.append(snapshot_version)
                clean_history.append(post_clean)
                poison_history.append(post_pois)
                asr_history.append(post_asr)
                recorded_versions.add(snapshot_version)
                post_unlearn_stop_version = snapshot_version + ASYNC_STOP_AFTER_UNLEARN_ROUNDS
                logging.info(
                    f"[SERVER] Post-unlearn evaluations logged; async stop target version set to {post_unlearn_stop_version}"
                )

                if not erased_client_removed:
                    erased_index = None
                    for idx, client in enumerate(clients):
                        if client.client_id == PARTY_TO_BE_ERASED:
                            client.stop()
                            erased_index = idx
                            break
                    if erased_index is not None:
                        clients.pop(erased_index)
                        server.deregister_client(PARTY_TO_BE_ERASED)
                        erased_client_removed = True
                        logging.info(f"[SERVER] Client {PARTY_TO_BE_ERASED} removed post-unlearning")

                if PAUSE_DURING_UNLEARN:
                    for client in clients:
                        client.resume()

                if not post_unlearn_adjustment_done:
                    for client in clients:
                        client.set_lr(POST_UNLEARN_LR)
                        client.set_local_training_config(num_local_epochs=1, num_updates_in_epoch=POST_UNLEARN_MAX_UPDATES)
                    post_unlearn_adjustment_done = True
                    logging.info(
                        f"[SERVER] Clients switched to post-unlearn LR={POST_UNLEARN_LR:.2e} and max_updates={POST_UNLEARN_MAX_UPDATES}"
                    )

                unlearn_triggered = True

            if current_version % ASYNC_EVAL_INTERVAL == 0 and current_version not in recorded_versions:
                snapshot_state, snapshot_version = server.get_snapshot()
                eval_model = create_model(num_classes)
                eval_model.load_state_dict(snapshot_state)
                clean_acc = Utils.eval_with_class_hist(testloader, eval_model, num_classes, device)
                pois_acc = Utils.evaluate(testloader_poison, eval_model)
                asr = compute_asr(eval_model, testloader_bd, target_label)
                logging.info(
                    f"[ASYNC-EVAL] version={snapshot_version} | Clean={clean_acc:.2f} | Backdoor={pois_acc:.2f} | ASR={asr:.2f}"
                )
                eval_versions.append(snapshot_version)
                clean_history.append(clean_acc)
                poison_history.append(pois_acc)
                asr_history.append(asr)
                recorded_versions.add(snapshot_version)

            if unlearn_triggered and post_unlearn_stop_version is not None and current_version >= post_unlearn_stop_version:
                logging.info(f"[SERVER] Reached post-unlearn stop version {post_unlearn_stop_version}; terminating training loop")
                break

    finally:
        for client in clients:
            client.stop()
        server.wait_for_update_queue()
        server.stop()

    fig = plt.figure(figsize=(8, 5))
    plt.plot(eval_versions, clean_history, "r-o", linewidth=2, label="Clean Acc")
    plt.plot(eval_versions, poison_history, "b-x", linewidth=2, label="Backdoor Acc (label flip)")
    plt.plot(eval_versions, asr_history, "g-^", linewidth=2, label="ASR")
    if unlearn_triggered and reference_version is not None:
        plt.axvline(x=reference_version + 1, linestyle="--", color="k", label="Unlearn event")
    plt.xlabel("Global Version")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    plot_filename = DOC_DIR / "images" / f"oasis_async_curve_{timestamp}.png"
    fig.savefig(plot_filename, dpi=150)
    logging.info(f"Plot saved to {plot_filename}")
    plt.close(fig)

    logging.info("Async OASIS run completed. Tuning tips: "
                 "increase UNLEARN_LR or UNLEARN_EPOCHS if ASR stays high; decrease them or shrink UNLEARN_PROJ_RADIUS if clean acc collapses.")


if __name__ == "__main__":
    run_async_oasis()

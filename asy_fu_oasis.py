import copy
import csv
import itertools
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from utils.model import FLNet, MRICNN
from utils.local_train import LocalTraining
from utils.utils import Utils
from utils.fusion import Fusion, FusionAvg, FusionRetrain

# -----------------------------------------------------------------------------
# Run-mode switch so that the async OASIS pipeline can be triggered without
# touching the legacy synchronous workflow.
# -----------------------------------------------------------------------------
RUN_MODE = os.environ.get('OASIS_RUN_MODE', 'sync').lower()
if RUN_MODE == 'async':
    from oasis_async_unlearning import run_async_oasis

    if __name__ == '__main__':
        run_async_oasis()
    else:
        run_async_oasis()
    sys.exit(0)

# -------------------------
# Repro & cuDNN / TF32
# -------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device is required for this experiment.")
device = torch.device('cuda')

doc_dir = Path(__file__).resolve().parent / 'doc'
doc_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# -------------------------
# Federated fusion selector
# -------------------------
def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):
    fusion_class_dict = {
        'FedAvg': FusionAvg(num_parties),
        'Retrain': FusionRetrain(num_parties),
        'Unlearn': FusionAvg(num_parties),
    }
    return fusion_class_dict[fusion_key]

# -------------------------
# Datasets
# -------------------------
class IndexedDataset(Dataset):
    """Wrap ImageFolder with fixed index list."""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, label = self.base_dataset[base_idx]
        return image, label
    def get_base_index(self, idx):
        return self.indices[idx]

class PoisonedIndexedDataset(IndexedDataset):
    """
    Adds a square patch trigger to selected indices and FLIPS label to target.
    Patch size is resolution-adaptive via patch_frac.
    """
    def __init__(self, base_dataset, indices, poison_indices, target_label,
                 patch_frac=24/256, trigger_value=1.0):
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
    """
    Only ADD trigger on NON-target samples; labels KEEP original.
    Used for ASR evaluation.
    """
    def __init__(self, base_dataset, indices, target_label,
                 patch_frac=24/256, trigger_value=1.0):
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

def compute_asr(model, loader, target_label, device):
    """Attack Success Rate on backdoor test set (labels not flipped)."""
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

# -------------------------
# Helpers
# -------------------------
def extract_patient_id(path: str) -> str:
    name = Path(path).stem
    parts = name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return parts[0]

def collect_indices(patient_ids, mapping):
    return list(itertools.chain.from_iterable(mapping[pid] for pid in patient_ids))

# -------------------------
# Dataset preparation
# -------------------------
num_parties = 5
party_to_be_erased = 0
percent_poison = 0.8
batch_size = 64

data_root = Path(__file__).resolve().parent / 'imagesoasis'
if not data_root.exists():
    raise FileNotFoundError(f'ImagesOASIS dataset not found at {data_root}.')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),   # 你已降到 128，保持
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

full_dataset = datasets.ImageFolder(root=str(data_root), transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# === 改1：后门目标改为“样本最少的类”，避免多数类假象 ===
targets_np = np.array(full_dataset.targets)
counts = np.bincount(targets_np, minlength=num_classes)
minority_class_idx = int(np.argmin(counts))
target_label = minority_class_idx
target_class_name = class_names[target_label]
print(f"Backdoor target set to MINORITY class: {target_class_name} (count={counts[target_label]})")

# 建 patient -> indices
patient_to_indices = {}
for idx, (path, _) in enumerate(full_dataset.samples):
    pid = extract_patient_id(path)
    patient_to_indices.setdefault(pid, []).append(idx)

patient_ids = list(patient_to_indices.keys())
rng = np.random.default_rng(SEED)
rng.shuffle(patient_ids)

train_split = max(num_parties, int(len(patient_ids) * 0.8))
train_split = min(train_split, len(patient_ids) - max(1, len(patient_ids) // 10))
train_split = max(train_split, num_parties)

train_patient_ids = patient_ids[:train_split]
test_patient_ids = patient_ids[train_split:]
if not test_patient_ids:
    test_patient_ids = train_patient_ids[-max(1, len(train_patient_ids) // 5):]
    train_patient_ids = train_patient_ids[:-len(test_patient_ids)]

train_patient_groups = np.array_split(np.array(train_patient_ids, dtype=object), num_parties)
party_indices_list = [collect_indices(group.tolist(), patient_to_indices) for group in train_patient_groups]

# 被擦除方：仅对“原本不是目标类”的样本投毒
erased_party_indices = party_indices_list[party_to_be_erased]
erased_candidates = [idx for idx in erased_party_indices if full_dataset.targets[idx] != target_label]
num_poison = int(len(erased_candidates) * percent_poison)
poison_indices = set(rng.choice(erased_candidates, size=num_poison, replace=False).tolist()) if num_poison > 0 else set()

# 组装各方 dataset
party_datasets = []
for party_id, indices in enumerate(party_indices_list):
    if party_id == party_to_be_erased:
        ds = PoisonedIndexedDataset(full_dataset, indices, poison_indices, target_label,
                                    patch_frac=24/256, trigger_value=1.0)
    else:
        ds = IndexedDataset(full_dataset, indices)
    party_datasets.append(ds)

# DataLoaders
trainloader_lst = [
    DataLoader(ds, batch_size=batch_size, shuffle=True,
               num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    for ds in party_datasets
]

test_patient_indices = collect_indices(test_patient_ids, patient_to_indices)
test_dataset = IndexedDataset(full_dataset, test_patient_indices)

# “标签翻转版”的后门测试（保留，便于和老指标对齐）
test_non_target_indices = [idx for idx in test_patient_indices if full_dataset.targets[idx] != target_label]
test_dataset_poison = PoisonedIndexedDataset(full_dataset, test_patient_indices, test_non_target_indices,
                                             target_label, patch_frac=24/256, trigger_value=1.0)

# “不翻标签”的 ASR 后门测试（推荐）
test_dataset_bd_keep_label = BackdoorTestDataset(full_dataset, test_patient_indices, target_label,
                                                 patch_frac=24/256, trigger_value=1.0)

testloader          = DataLoader(test_dataset, batch_size=128, shuffle=False,
                                 num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
testloader_poison   = DataLoader(test_dataset_poison, batch_size=128, shuffle=False,
                                 num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
testloader_bd_asr   = DataLoader(test_dataset_bd_keep_label, batch_size=128, shuffle=False,
                                 num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)

print(f'Classes: {class_names}')
print(f'Total patients: {len(patient_ids)} | train: {len(train_patient_ids)} | test: {len(test_patient_ids)}')
for idx, group in enumerate(train_patient_groups):
    print(f'Party {idx}: {len(group)} patients, {len(party_indices_list[idx])} slices')
print(f'Poison samples selected for party {party_to_be_erased}: {len(poison_indices)} -> target "{target_class_name}"')

# === 改2：全局类别权重（缓解极不平衡） ===
class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * num_classes  # 归一化到平均权重=1
class_weights = class_weights.to(device)

def create_model():
    """Factory for the federated model."""
    return MRICNN(num_classes).to(device)

# -------------------------
# Pre-FL training
# -------------------------
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
asr_metric = {}  # 新增：ASR
metrics_history = []

for fusion_key in fusion_types:
    loss_fed[fusion_key] = np.zeros(num_fl_rounds)
    clean_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    pois_accuracy[fusion_key] = np.zeros(num_fl_rounds)
    asr_metric[fusion_key]  = np.zeros(num_fl_rounds)
    if fusion_key != 'Retrain':
        dist_Retrain[fusion_key] = np.zeros(num_fl_rounds)

party_models_dict = {}
initial_model = create_model()
model_dict = {fusion_key: copy.deepcopy(initial_model.state_dict()) for fusion_key in fusion_types}

for round_num in range(num_fl_rounds):
    local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)

    for fusion_key in fusion_types:
        fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)

        current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
        current_model = create_model()
        current_model.load_state_dict(current_model_state_dict)

        party_models = []
        party_losses = []
        for party_id in range(num_parties):
            if fusion_key == 'Retrain' and party_id == party_to_be_erased:
                party_models.append(create_model())
                continue

            model = copy.deepcopy(current_model)
            # === 改3：把“带权交叉熵”传给本地训练，缓解不平衡 ===
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            model_update, party_loss = local_training.train(
                model=model,
                trainloader=trainloader_lst[party_id],
                criterion=criterion,
                opt=None,
            )
            party_models.append(copy.deepcopy(model_update))
            party_losses.append(party_loss)

        loss_fed[fusion_key][round_num] += np.mean(party_losses) / num_of_repeats

        current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)
        model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
        party_models_dict[fusion_key] = party_models

        eval_model = create_model()
        eval_model.load_state_dict(current_model_state_dict)
        clean_acc = Utils.evaluate(testloader, eval_model)
        pois_acc  = Utils.evaluate(testloader_poison, eval_model)  # 旧“翻标签”口径（保留）
        asr       = compute_asr(eval_model, testloader_bd_asr, target_label, device)  # 新：ASR

        clean_accuracy[fusion_key][round_num] = clean_acc
        pois_accuracy[fusion_key][round_num]  = pois_acc
        asr_metric[fusion_key][round_num]     = asr

        metrics_history.append(
            {
                'phase': 'pre_unlearn',
                'fusion': fusion_key,
                'round': round_num,
                'clean_acc': clean_acc,
                'pois_acc': pois_acc,
                'asr': asr,
            }
        )

        print(f'Global Clean Acc {fusion_key}, round {round_num} = {clean_acc:.2f}')
        print(f'Global Poison(FlipLabel) Acc {fusion_key}, round {round_num} = {pois_acc:.2f}')
        print(f'Global ASR {fusion_key}, round {round_num} = {asr:.2f}')

# quick final before unlearn
for fusion_key in fusion_types:
    current_model_state_dict = model_dict[fusion_key]
    current_model = create_model()
    current_model.load_state_dict(current_model_state_dict)
    clean_acc = Utils.evaluate(testloader, current_model)
    pois_acc  = Utils.evaluate(testloader_poison, current_model)
    asr       = compute_asr(current_model, testloader_bd_asr, target_label, device)

    metrics_history.append(
        {
            'phase': 'pre_unlearn_final',
            'fusion': fusion_key,
            'round': None,
            'clean_acc': clean_acc,
            'pois_acc': pois_acc,
            'asr': asr,
        }
    )

    print(f'[Final before Unlearn] {fusion_key}: Clean={clean_acc:.2f} | Poison(FlipLabel)={pois_acc:.2f} | ASR={asr:.2f}')

# -------------------------
# Unlearning (projected ascent on erased party)
# -------------------------
num_updates_in_epoch = 200
num_local_epochs_unlearn = 5
lr = 0.01
distance_threshold = 2.2
clip_grad = 5

initial_model = create_model()
unlearned_model_dict = {}
for fusion_key in fusion_types_unlearn:
    if fusion_key == 'Retrain':
        unlearned_model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

clean_accuracy_unlearn = {fusion_key: 0 for fusion_key in fusion_types_unlearn}
pois_accuracy_unlearn  = {fusion_key: 0 for fusion_key in fusion_types_unlearn}

for fusion_key in fusion_types:
    if fusion_key == 'Retrain':
        continue

    initial_model = create_model()
    fedavg_model_state_dict = copy.deepcopy(model_dict[fusion_key])
    fedavg_model = create_model()
    fedavg_model.load_state_dict(fedavg_model_state_dict)

    party_models = copy.deepcopy(party_models_dict[fusion_key])
    party0_model = copy.deepcopy(party_models[party_to_be_erased])

    model_ref_vec = (
        num_parties / (num_parties - 1) * nn.utils.parameters_to_vector(fedavg_model.parameters())
        - 1 / (num_parties - 1) * nn.utils.parameters_to_vector(party0_model.parameters())
    )
    model_ref = create_model()
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    eval_model = copy.deepcopy(model_ref)
    clean_ref = Utils.evaluate(testloader, eval_model)
    pois_ref = Utils.evaluate(testloader_poison, eval_model)
    asr_ref = compute_asr(eval_model, testloader_bd_asr, target_label, device)

    metrics_history.append(
        {
            'phase': 'reference_model',
            'fusion': fusion_key,
            'round': None,
            'clean_acc': clean_ref,
            'pois_acc': pois_ref,
            'asr': asr_ref,
        }
    )

    print(f'Clean Acc for Reference Model = {clean_ref:.2f}')
    print(f'Poison(FlipLabel) Acc for Reference Model = {pois_ref:.2f}')
    print(f'ASR   for Reference Model = {asr_ref:.2f}')

    dist_ref_random_lst = [Utils.get_distance(model_ref, create_model()) for _ in range(10)]
    threshold = np.mean(dist_ref_random_lst) / 3
    print(f'Radius (proj) for model_ref: {threshold:.4f}')
    print(f'Distance(model_ref, party0) = {Utils.get_distance(model_ref, party0_model):.4f}')

    # Unlearn on erased party via gradient ascent with projection
    model = copy.deepcopy(model_ref)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    flag = False
    for epoch in range(num_local_epochs_unlearn):
        print('------------', epoch)
        if flag: break
        for batch_id, (x_batch, y_batch) in enumerate(trainloader_lst[party_to_be_erased]):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            opt.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_joint = -loss  # gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            with torch.no_grad():
                distance = Utils.get_distance(model, model_ref)
                if distance > threshold and distance > 0:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(model_ref.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * math.sqrt(threshold)
                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())

            distance_ref_party_0 = Utils.get_distance(model, party0_model)
            print('Distance from the unlearned model to party 0:', distance_ref_party_0)
            if distance_ref_party_0 > distance_threshold:
                flag = True
                break
            if num_updates_in_epoch is not None and batch_id >= num_updates_in_epoch:
                break

    unlearned_model = copy.deepcopy(model)
    unlearned_model_dict[fusion_types_unlearn[1]] = unlearned_model.state_dict()

    eval_model = create_model()
    eval_model.load_state_dict(unlearned_model_dict[fusion_types_unlearn[1]])
    unlearn_clean_acc = Utils.evaluate(testloader, eval_model)
    unlearn_pois_acc = Utils.evaluate(testloader_poison, eval_model)
    unlearn_asr       = compute_asr(eval_model, testloader_bd_asr, target_label, device)

    metrics_history.append(
        {
            'phase': 'post_unlearn_init',
            'fusion': fusion_types_unlearn[1],
            'round': None,
            'clean_acc': unlearn_clean_acc,
            'pois_acc': unlearn_pois_acc,
            'asr': unlearn_asr,
        }
    )

    print(f'UN-Local: Clean={unlearn_clean_acc:.2f} | Poison(FlipLabel)={unlearn_pois_acc:.2f} | ASR={unlearn_asr:.2f}')
    clean_accuracy_unlearn[fusion_types_unlearn[1]] = unlearn_clean_acc

# -------------------------
# FL after unlearning
# -------------------------
num_fl_after_unlearn_rounds = num_fl_rounds
num_updates_in_epoch = 50
num_local_epochs = 1

clean_accuracy_unlearn_fl_after_unlearn = {fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn}
pois_accuracy_unlearn_fl_after_unlearn  = {fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn}
asr_unlearn_fl_after_unlearn            = {fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn}
loss_unlearn = {fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn}

for round_num in range(num_fl_after_unlearn_rounds):
    local_training = LocalTraining(num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs)
    for fusion_key in fusion_types_unlearn:
        fusion = FL_round_fusion_selection(num_parties=num_parties - 1, fusion_key=fusion_key)

        current_model_state_dict = copy.deepcopy(unlearned_model_dict[fusion_key])
        current_model = create_model()
        current_model.load_state_dict(current_model_state_dict)

        party_models = []
        party_losses = []
        for party_id in range(1, num_parties):  # erased party removed
            model = copy.deepcopy(current_model)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            model_update, party_loss = local_training.train(
                model=model, trainloader=trainloader_lst[party_id],
                criterion=criterion, opt=None,
            )
            party_models.append(copy.deepcopy(model_update))
            party_losses.append(party_loss)

        loss_unlearn[fusion_key][round_num] = np.mean(party_losses)
        current_model_state_dict = fusion.fusion_algo(party_models=party_models, current_model=current_model)
        unlearned_model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
        party_models_dict[fusion_key] = party_models

        eval_model = create_model()
        eval_model.load_state_dict(current_model_state_dict)
        clean_acc = Utils.evaluate(testloader, eval_model)
        pois_acc  = Utils.evaluate(testloader_poison, eval_model)
        asr       = compute_asr(eval_model, testloader_bd_asr, target_label, device)

        clean_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = clean_acc
        pois_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num]  = pois_acc
        asr_unlearn_fl_after_unlearn[fusion_key][round_num]            = asr

        metrics_history.append(
            {
                'phase': 'post_unlearn',
                'fusion': fusion_key,
                'round': round_num,
                'clean_acc': clean_acc,
                'pois_acc': pois_acc,
                'asr': asr,
            }
        )

        print(f'Global Clean Acc {fusion_key}, round {round_num} = {clean_acc:.2f}')
        print(f'Global Poison(FlipLabel) Acc {fusion_key}, round {round_num} = {pois_acc:.2f}')
        print(f'Global ASR {fusion_key}, round {round_num} = {asr:.2f}')

# -------------------------
# Persist metrics
# -------------------------
csv_path = doc_dir / f'training_metrics_{timestamp}.csv'
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['phase', 'fusion', 'round', 'clean_acc', 'pois_acc', 'asr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in metrics_history:
        row = entry.copy()
        row['round'] = row['round'] if row['round'] is not None else ''
        writer.writerow(row)
print(f'Metrics log saved to {csv_path}')

# -------------------------
# Plots (以 ASR 为主；保留原 Poison Acc 便于对照)
# -------------------------
fl_rounds = list(range(1, num_fl_rounds + 1))
fig_asr = plt.figure()
plt.plot(fl_rounds, asr_unlearn_fl_after_unlearn['Unlearn'], 'ro--', linewidth=2, markersize=8, label='UN-ASR')
plt.plot(fl_rounds, asr_unlearn_fl_after_unlearn['Retrain'], 'm^-', linewidth=2, markersize=8, label='Retrain-ASR')
plt.xlabel('Training Rounds')
plt.ylabel('ASR (%)')
plt.grid(True)
plt.xlim([1, num_fl_rounds])
plt.legend()
fig_asr.tight_layout()
asr_fig_path = doc_dir / f'asr_comparison_{timestamp}.png'
fig_asr.savefig(asr_fig_path, dpi=150)
print(f'ASR plot saved to {asr_fig_path}')
plt.show()

fig_clean = plt.figure()
plt.plot(fl_rounds, clean_accuracy_unlearn_fl_after_unlearn['Unlearn'], 'ro--', linewidth=2, markersize=8, label='UN-Clean Acc')
plt.plot(fl_rounds, clean_accuracy_unlearn_fl_after_unlearn['Retrain'], 'm^-', linewidth=2, markersize=8, label='Retrain-Clean Acc')
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xlim([1, num_fl_rounds])
plt.legend()
fig_clean.tight_layout()
clean_fig_path = doc_dir / f'clean_accuracy_{timestamp}.png'
fig_clean.savefig(clean_fig_path, dpi=150)
print(f'Clean accuracy plot saved to {clean_fig_path}')
plt.show()

fig_pois = plt.figure()
plt.plot(fl_rounds, pois_accuracy_unlearn_fl_after_unlearn['Unlearn'], 'gx--', linewidth=2, markersize=8, label='UN-Poison(FlipLabel) Acc')
plt.plot(fl_rounds, pois_accuracy_unlearn_fl_after_unlearn['Retrain'], 'c+-', linewidth=2, markersize=8, label='Retrain-Poison(FlipLabel) Acc')
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xlim([1, num_fl_rounds])
plt.legend()
fig_pois.tight_layout()
pois_fig_path = doc_dir / f'poison_accuracy_{timestamp}.png'
fig_pois.savefig(pois_fig_path, dpi=150)
print(f'Poison accuracy plot saved to {pois_fig_path}')
plt.show()

import logging
import copy
import itertools
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from utils.fusion import FusionAvg, FusionRetrain
from utils.local_train import LocalTraining
from utils.model import OASISNet
from utils.utils import Utils

# -----------------------------
# Seeding and device selection
# -----------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Logging configuration
# -----------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "doc" / "logs"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "doc" / "images"
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

log_file_name = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = LOG_DIR / log_file_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)

# MNIST script uses CUDA directly; keep the behaviour but fall back to CPU when
# unavailable so the script can still run in constrained environments.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Helper functions
# -----------------------------

def FL_round_fusion_selection(num_parties, fusion_key="FedAvg"):
    """Mirror the selector from fu_mnist.py."""

    fusion_class_dict = {
        "FedAvg": FusionAvg(num_parties),
        "Retrain": FusionRetrain(num_parties),
        "Unlearn": FusionAvg(num_parties),
    }

    return fusion_class_dict[fusion_key]


def add_oasis_trigger(img_tensor: torch.Tensor) -> torch.Tensor:
    """Add a visible L-shaped trigger patch to a 3×H×W tensor in [0, 1]."""

    img = img_tensor.clone()
    h, w = img.shape[1], img.shape[2]
    patch_size = 16
    y_start = h - patch_size
    x_start = w - patch_size

    color = torch.tensor([1.0, 0.0, 0.0], device=img.device).view(3, 1, 1)
    img[:, y_start:, x_start : x_start + 3] = color  # vertical bar
    img[:, y_start : y_start + 3, x_start:] = color  # horizontal bar
    return img


def _load_balanced_oasis_images(base_dir: Path, max_per_class: int, image_size: int):
    """Load and down-sample OASIS data to a balanced subset."""

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
    # (N, 128, 128, 3) -> (N, 3, 128, 128)
    images = np.transpose(images, (0, 3, 1, 2))
    labels = np.array(labels, dtype=np.int64)
    return images, labels


class OASISAugmentedDataset(Dataset):
    """Tensor-backed dataset with optional on-the-fly augmentation."""

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


def _dirichlet_split_indices(y_train: np.ndarray, num_parties: int, alpha: float):
    """Generate non-IID client splits using Dirichlet label skew."""

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

    for party_indices in indices_per_party:
        random.shuffle(party_indices)
    return indices_per_party


def _make_party_loader(
    x_party: np.ndarray,
    y_party: np.ndarray,
    batch_size: int,
    target_label: int,
    party_id: int,
    party_to_be_erased: int,
    poison_ratio: float,
    train_transform,
):
    """Create a party dataloader with optional backdoor injection and reweighting."""

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
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(), num_samples=len(sample_weights), replacement=True
    )

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
    """
    Load OASIS data, build a balanced subset, perform Dirichlet label-skew split,
    inject backdoor samples for the erased party, and prepare clean/poison test loaders.
    """

    images, labels = _load_balanced_oasis_images(
        base_dir=base_dir, max_per_class=max_per_class, image_size=image_size
    )

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.15, stratify=labels, random_state=SEED
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

    # Clean test loader
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    )
    testloader_clean = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Poisoned test loader: trigger on all samples with target_label
    poisoned_test_x = torch.tensor(x_test, dtype=torch.float32)
    poisoned_test_y = torch.full_like(torch.tensor(y_test, dtype=torch.long), target_label)
    for idx in range(len(poisoned_test_x)):
        poisoned_test_x[idx] = add_oasis_trigger(poisoned_test_x[idx])
    poisoned_test_dataset = TensorDataset(poisoned_test_x, poisoned_test_y)
    testloader_poison = DataLoader(poisoned_test_dataset, batch_size=64, shuffle=False)

    return trainloader_lst, testloader_clean, testloader_poison


# -----------------------------
# Experiment configuration
# -----------------------------
num_parties = 5
party_to_be_erased = 0
base_data_dir = Path(__file__).resolve().parent / "imagesoasis"  # Adjust to the actual dataset location
image_size = 128
max_per_class = 2000  # increase data per class to stabilise training
alpha = 1.0  # non-IID strength; increase toward IID for debugging
batch_size = 32
num_of_repeats = 1
num_fl_rounds = 30
num_local_epochs = 2
num_updates_in_epoch = None
base_lr = 5e-3

# Backdoor configuration
poison_ratio = 0.6  # softer backdoor ratio to preserve clean accuracy
target_label = 0

# Training/Unlearning hyperparameters
# Unlearning settings use a slightly larger LR and more updates to ensure
# meaningful gradient ascent against the erased client's data on OASIS.
num_local_epochs_unlearn = 2  # run at least one full epoch of ascent (2 by default)
num_updates_in_epoch_unlearn = 50  # cap batches per epoch during unlearning to ensure tens of updates
unlearning_lr = 2e-3  # larger than base LR to make backdoor forgetting effective on OASIS
unlearn_distance_factor = 3.0  # adaptive early-stop: allow the model to move several times the initial ref distance
clip_grad = 1.0
num_fl_after_unlearn_rounds = 10
num_updates_in_epoch_after = 50
num_local_epochs_after = 1
after_unlearn_lr = 2.5e-3


# -----------------------------
# Data preparation
# -----------------------------
print("Preparing OASIS federated loaders...")
logging.info("Preparing OASIS federated loaders...")
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
logging.info("Data preparation complete.")


# -----------------------------
# Phase 1: standard FL training (FedAvg + Retrain)
# -----------------------------
fusion_types = ["FedAvg", "Retrain"]
fusion_types_unlearn = ["Retrain", "Unlearn"]

loss_fed = {fusion_key: np.zeros(num_fl_rounds) for fusion_key in fusion_types}
clean_accuracy = {fusion_key: np.zeros(num_fl_rounds) for fusion_key in fusion_types}
pois_accuracy = {fusion_key: np.zeros(num_fl_rounds) for fusion_key in fusion_types}
dist_Retrain = {fusion_key: np.zeros(num_fl_rounds) for fusion_key in fusion_types if fusion_key != "Retrain"}

party_models_dict = {}
initial_model = OASISNet().to(device)
model_dict = {fusion_key: copy.deepcopy(initial_model.state_dict()) for fusion_key in fusion_types}

for round_num in range(num_fl_rounds):
    local_training = LocalTraining(
        num_updates_in_epoch=num_updates_in_epoch, num_local_epochs=num_local_epochs
    )

    for fusion_key in fusion_types:
        fusion = FL_round_fusion_selection(num_parties=num_parties, fusion_key=fusion_key)

        current_model_state_dict = copy.deepcopy(model_dict[fusion_key])
        current_model = copy.deepcopy(initial_model)
        current_model.load_state_dict(current_model_state_dict)

        # --------------------- Local Training Round ---------------------
        party_models = []
        party_losses = []
        for party_id in range(num_parties):
            if fusion_key == "Retrain" and party_id == party_to_be_erased:
                party_models.append(OASISNet().to(device))
            else:
                model = copy.deepcopy(current_model)
                model_update, party_loss = local_training.train(
                    model=model,
                    trainloader=trainloader_lst[party_id],
                    criterion=None,
                    opt=None,
                    lr=base_lr,
                )

                party_models.append(copy.deepcopy(model_update))
                party_losses.append(party_loss)

        if len(party_losses) > 0:
            loss_fed[fusion_key][round_num] += np.mean(party_losses) / num_of_repeats
        # ----------------------------------------------------------------

        current_model_state_dict = fusion.fusion_algo(
            party_models=party_models, current_model=current_model
        )
        model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
        party_models_dict[fusion_key] = party_models

        eval_model = OASISNet().to(device)
        eval_model.load_state_dict(current_model_state_dict)
        clean_acc = Utils.evaluate(testloader_clean, eval_model)
        clean_accuracy[fusion_key][round_num] = clean_acc
        print(f"Global Clean Accuracy {fusion_key}, round {round_num} = {clean_acc}")
        logging.info(f"Global Clean Accuracy {fusion_key}, round {round_num} = {clean_acc}")
        pois_acc = Utils.evaluate(testloader_poison, eval_model)
        pois_accuracy[fusion_key][round_num] = pois_acc
        print(f"Global Backdoor Accuracy {fusion_key}, round {round_num} = {pois_acc}")
        logging.info(f"Global Backdoor Accuracy {fusion_key}, round {round_num} = {pois_acc}")

for fusion_key in fusion_types:
    current_model_state_dict = model_dict[fusion_key]
    current_model = copy.deepcopy(initial_model)
    current_model.load_state_dict(current_model_state_dict)
    clean_acc = Utils.evaluate(testloader_clean, current_model)
    print(f"Clean Accuracy {fusion_key}: {clean_acc}")
    logging.info(f"Clean Accuracy {fusion_key}: {clean_acc}")
    pois_acc = Utils.evaluate(testloader_poison, current_model)
    print(f"Backdoor Accuracy {fusion_key}: {pois_acc}")
    logging.info(f"Backdoor Accuracy {fusion_key}: {pois_acc}")


# -----------------------------
# Phase 2: asynchronous unlearning (Halimi-style gradient ascent)
# -----------------------------
unlearned_model_dict = {}
for fusion_key in fusion_types_unlearn:
    if fusion_key == "Retrain":
        unlearned_model_dict[fusion_key] = copy.deepcopy(initial_model.state_dict())

clean_accuracy_unlearn = {fusion_key: 0 for fusion_key in fusion_types_unlearn}
pois_accuracy_unlearn = {fusion_key: 0 for fusion_key in fusion_types_unlearn}

for fusion_key in fusion_types:
    if fusion_key == "Retrain":
        continue

    initial_model = OASISNet().to(device)
    fedavg_model_state_dict = copy.deepcopy(model_dict[fusion_key])
    fedavg_model = copy.deepcopy(initial_model)
    fedavg_model.load_state_dict(fedavg_model_state_dict)

    party_models = copy.deepcopy(party_models_dict[fusion_key])
    party0_model = copy.deepcopy(party_models[party_to_be_erased])

    w_fed = nn.utils.parameters_to_vector(fedavg_model.parameters())
    w_party0 = nn.utils.parameters_to_vector(party0_model.parameters())

    # Compute reference model: w_ref = N/(N-1) * w^T - 1/(N-1) * w^{T-1}_i
    model_ref_vec = (num_parties / (num_parties - 1)) * w_fed - (1.0 / (num_parties - 1)) * w_party0

    # Compute threshold
    model_ref = copy.deepcopy(fedavg_model)
    nn.utils.vector_to_parameters(model_ref_vec, model_ref.parameters())

    eval_model = copy.deepcopy(model_ref)
    unlearn_clean_acc = Utils.evaluate(testloader_clean, eval_model)
    print(f"Clean Accuracy for Reference Model = {unlearn_clean_acc}")
    logging.info(f"Clean Accuracy for Reference Model = {unlearn_clean_acc}")
    unlearn_pois_acc = Utils.evaluate(testloader_poison, eval_model)
    print(f"Backdoor Accuracy for Reference Model = {unlearn_pois_acc}")
    logging.info(f"Backdoor Accuracy for Reference Model = {unlearn_pois_acc}")

    dist_ref_random_lst = []
    for _ in range(10):
        dist_ref_random_lst.append(Utils.get_distance(model_ref, OASISNet().to(device)))

    print(f"Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}")
    logging.info(f"Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}")
    threshold = np.mean(dist_ref_random_lst) / 3
    print(f"Radius for model_ref: {threshold}")
    logging.info(f"Radius for model_ref: {threshold}")
    dist_ref_party0_init = Utils.get_distance(model_ref, party0_model)
    print(f"Distance of Reference Model to party0_model: {dist_ref_party0_init}")
    logging.info(
        f"Distance of Reference Model to party0_model: {dist_ref_party0_init}"
    )

    # Adaptive early-stop threshold based on initial distance to the erased party.
    distance_threshold = dist_ref_party0_init * unlearn_distance_factor
    logging.info(
        f"Initial distance model_ref <-> party0 = {dist_ref_party0_init}"
    )
    logging.info(f"Unlearning distance_threshold = {distance_threshold}")

    # --------------------- Unlearning ---------------------
    model = copy.deepcopy(model_ref)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=unlearning_lr, momentum=0.9)

    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    flag = False
    for epoch in range(num_local_epochs_unlearn):
        print("------------", epoch)
        logging.info(f"------------ Unlearning Epoch {epoch} ------------")
        if flag:
            break
        for batch_id, (x_batch, y_batch) in enumerate(trainloader_lst[party_to_be_erased]):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss_joint = -loss  # negate the loss for gradient ascent
            loss_joint.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            with torch.no_grad():
                distance = Utils.get_distance(model, model_ref)
                if distance > threshold:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()) - nn.utils.parameters_to_vector(
                        model_ref.parameters()
                    )
                    dist_vec = dist_vec / torch.norm(dist_vec) * math.sqrt(threshold)
                    proj_vec = nn.utils.parameters_to_vector(model_ref.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    distance = Utils.get_distance(model, model_ref)

            distance_ref_party_0 = Utils.get_distance(model, party0_model)
            print("Distance from the unlearned model to party 0:", distance_ref_party_0)
            logging.info(f"Distance from the unlearned model to party 0: {distance_ref_party_0}")

            if distance_ref_party_0 > distance_threshold:
                flag = True
                break

            if num_updates_in_epoch_unlearn is not None and batch_id >= num_updates_in_epoch_unlearn:
                break
    # ------------------------------------------------------

    unlearned_model = copy.deepcopy(model)
    unlearned_model_dict[fusion_types_unlearn[1]] = unlearned_model.state_dict()

    eval_model = OASISNet().to(device)
    eval_model.load_state_dict(unlearned_model_dict[fusion_types_unlearn[1]])
    unlearn_clean_acc = Utils.evaluate(testloader_clean, eval_model)
    print(f"Clean Accuracy for UN-Local Model = {unlearn_clean_acc}")
    logging.info(f"Clean Accuracy for UN-Local Model = {unlearn_clean_acc}")
    clean_accuracy_unlearn[fusion_types_unlearn[1]] = unlearn_clean_acc
    pois_unlearn_acc = Utils.evaluate(testloader_poison, eval_model)
    print(f"Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}")
    logging.info(f"Backdoor Accuracy for UN-Local Model = {pois_unlearn_acc}")
    pois_accuracy_unlearn[fusion_types_unlearn[1]] = pois_unlearn_acc


# -----------------------------
# Phase 3: continue FL after unlearning (Retrain vs Unlearn)
# -----------------------------
clean_accuracy_unlearn_fl_after_unlearn = {
    fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn
}
pois_accuracy_unlearn_fl_after_unlearn = {
    fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn
}
loss_unlearn = {fusion_key: np.zeros(num_fl_after_unlearn_rounds) for fusion_key in fusion_types_unlearn}

for round_num in range(num_fl_after_unlearn_rounds):
    local_training = LocalTraining(
        num_updates_in_epoch=num_updates_in_epoch_after, num_local_epochs=num_local_epochs_after
    )

    for fusion_key in fusion_types_unlearn:
        fusion_num_parties = num_parties if fusion_key == "Retrain" else num_parties - 1
        fusion = FL_round_fusion_selection(num_parties=fusion_num_parties, fusion_key=fusion_key)

        current_model_state_dict = copy.deepcopy(unlearned_model_dict[fusion_key])
        current_model = OASISNet().to(device)
        current_model.load_state_dict(current_model_state_dict)

        # --------------------- Local Training Round ---------------------
        party_models = []
        party_losses = []
        for party_id in range(1, num_parties):
            model = copy.deepcopy(current_model)
            model_update, party_loss = local_training.train(
                model=model,
                trainloader=trainloader_lst[party_id],
                criterion=None,
                opt=None,
                lr=after_unlearn_lr,
            )

            party_models.append(copy.deepcopy(model_update))
            party_losses.append(party_loss)

        if len(party_losses) > 0:
            loss_unlearn[fusion_key][round_num] = np.mean(party_losses)
        # ----------------------------------------------------------------

        current_model_state_dict = fusion.fusion_algo(
            party_models=party_models, current_model=current_model
        )
        unlearned_model_dict[fusion_key] = copy.deepcopy(current_model_state_dict)
        party_models_dict[fusion_key] = party_models

        eval_model = OASISNet().to(device)
        eval_model.load_state_dict(current_model_state_dict)
        unlearn_clean_acc = Utils.evaluate(testloader_clean, eval_model)
        print(f"Global Clean Accuracy {fusion_key}, round {round_num} = {unlearn_clean_acc}")
        logging.info(f"Global Clean Accuracy {fusion_key}, round {round_num} = {unlearn_clean_acc}")
        clean_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_clean_acc
        unlearn_pois_acc = Utils.evaluate(testloader_poison, eval_model)
        print(f"Global Backdoor Accuracy {fusion_key}, round {round_num} = {unlearn_pois_acc}")
        logging.info(f"Global Backdoor Accuracy {fusion_key}, round {round_num} = {unlearn_pois_acc}")
        pois_accuracy_unlearn_fl_after_unlearn[fusion_key][round_num] = unlearn_pois_acc


# -----------------------------
# Plotting
# -----------------------------
fl_rounds = [i for i in range(1, num_fl_after_unlearn_rounds + 1)]

plt.plot(
    fl_rounds,
    clean_accuracy_unlearn_fl_after_unlearn["Unlearn"],
    "ro--",
    linewidth=2,
    markersize=12,
    label="UN-Clean Acc",
)
plt.plot(
    fl_rounds,
    pois_accuracy_unlearn_fl_after_unlearn["Unlearn"],
    "gx--",
    linewidth=2,
    markersize=12,
    label="UN-Backdoor Acc",
)
plt.plot(
    fl_rounds,
    clean_accuracy_unlearn_fl_after_unlearn["Retrain"],
    "m^-",
    linewidth=2,
    markersize=12,
    label="Retrain-Clean Acc",
)
plt.plot(
    fl_rounds,
    pois_accuracy_unlearn_fl_after_unlearn["Retrain"],
    "c+-",
    linewidth=2,
    markersize=12,
    label="Retrain-Backdoor Acc",
)
plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("OASIS Federated Unlearning: Retrain vs Unlearn")
plt.grid()
plt.ylim([0, 100])
plt.xlim([1, num_fl_after_unlearn_rounds])
plt.legend()

image_file_name = f"unlearning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
image_file_path = IMAGE_DIR / image_file_name
plt.savefig(image_file_path)
logging.info(f"Plot saved to {image_file_path}")
plt.show()

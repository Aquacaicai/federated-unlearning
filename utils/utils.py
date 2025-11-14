import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Utils():
    
    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance.item()

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = Utils.get_distance(current_model, party_models[i])
        return distances

    def evaluate(testloader, model):
        model.eval()
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    @staticmethod
    def eval_with_class_hist(loader, model, num_classes, device):
        model.eval()
        total = 0
        correct = 0
        pred_hist = np.zeros(num_classes, dtype=int)
        true_hist = np.zeros(num_classes, dtype=int)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
                for c in range(num_classes):
                    pred_hist[c] += (preds == c).sum().item()
                    true_hist[c] += (y == c).sum().item()
        acc = 100.0 * correct / max(1, total)
        logging.info(
            f"[DEBUG] Acc={acc:.2f} True={true_hist.tolist()} Pred={pred_hist.tolist()}"
        )
        return acc




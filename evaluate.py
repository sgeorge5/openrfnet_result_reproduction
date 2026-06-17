import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import load_pkl_dataset, split_known_unknown_classes, build_datasets
from preprocess import STFTTransform
from model import OpenRFNet
from train_openmax import openmax_score


# ---------------------------------------------------------
# Plot confusion matrix
# ---------------------------------------------------------
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Closed-set evaluation
# ---------------------------------------------------------
def evaluate_closed_set(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, labels, _ in tqdm(loader, desc="Closed-set evaluation"):
            x, labels = x.to(device), labels.to(device)
            _, logits = model(x)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)

    return accuracy, cm, report


# ---------------------------------------------------------
# Open-set evaluation (OpenMax)
# ---------------------------------------------------------
def evaluate_open_set(model, loader, device, class_centers, weibull_models, known_classes):
    model.eval()

    correct_known = 0
    correct_unknown = 0
    total_known = 0
    total_unknown = 0

    with torch.no_grad():
        for x, labels, class_names in tqdm(loader, desc="Open-set evaluation"):
            x = x.to(device)
            fused_feat, _, logits = model(x, return_features=True)

            fused_feat = fused_feat.cpu().numpy()
            logits = logits.cpu().numpy()

            for i in range(len(labels)):
                feature = fused_feat[i]
                class_name = class_names[i]

                # Compute OpenMax unknown probability
                unk_prob = openmax_score(feature, class_centers, weibull_models)

                if class_name in known_classes:
                    total_known += 1
                    pred = np.argmax(logits[i])
                    if pred == labels[i].item() and unk_prob < 0.5:
                        correct_known += 1
                else:
                    total_unknown += 1
                    if unk_prob >= 0.5:
                        correct_unknown += 1

    known_acc = correct_known / total_known
    unknown_acc = correct_unknown / total_unknown
    open_set_acc = (correct_known + correct_unknown) / (total_known + total_unknown)

    return known_acc, unknown_acc, open_set_acc


# ---------------------------------------------------------
# Main evaluation script
# ---------------------------------------------------------
def main():
    
    DATA_PATH = "RML2016data.pkl"
    MODEL_PATH = "best_model.pth"
    WEIBULL_PATH = "weibull_params.npy"
    CENTERS_PATH = "class_centers.npy"
    BATCH_SIZE = 32
    KNOWN_RATIO = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load dataset
    # -----------------------------
    data_dict = load_pkl_dataset(DATA_PATH)
    all_classes = list(data_dict.keys())

    known_classes, unknown_classes = split_known_unknown_classes(
        all_classes, known_ratio=KNOWN_RATIO
    )

    train_dataset, test_dataset = build_datasets(
        data_dict, known_classes, unknown_classes
    )

    transform = STFTTransform()
    train_dataset.transform = transform
    test_dataset.transform = transform

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Load model
    # -----------------------------
    model = OpenRFNet(num_classes=len(known_classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded closed-set model.")

    # -----------------------------
    # Load OpenMax parameters
    # -----------------------------
    weibull_models = np.load(WEIBULL_PATH, allow_pickle=True).item()
    class_centers = np.load(CENTERS_PATH, allow_pickle=True).item()
    print("Loaded OpenMax parameters.")

    # -----------------------------
    # Closed-set evaluation
    # -----------------------------
    print("\n===== CLOSED-SET EVALUATION =====")
    closed_acc, cm, report = evaluate_closed_set(
        model, test_loader, device, known_classes
    )

    print(f"Closed-set accuracy: {closed_acc:.4f}")
    print(report)
    plot_confusion_matrix(cm, known_classes)

    # -----------------------------
    # Open-set evaluation
    # -----------------------------
    print("\n===== OPEN-SET EVALUATION =====")
    known_acc, unknown_acc, open_set_acc = evaluate_open_set(
        model, test_loader, device, class_centers, weibull_models, known_classes
    )

    print(f"Known-class accuracy:   {known_acc:.4f}")
    print(f"Unknown detection rate: {unknown_acc:.4f}")
    print(f"Open-set accuracy:      {open_set_acc:.4f}")


if __name__ == "__main__":
    main()

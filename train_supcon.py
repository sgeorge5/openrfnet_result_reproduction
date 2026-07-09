import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os

from dataset import load_pkl_dataset, split_known_unknown_classes, build_datasets
from stft_transform import STFTTransform
from model import OpenRFNet
from losses import CombinedLoss


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_ce, total_supcon = 0, 0, 0

    for x, labels, _ in tqdm(loader, desc="Training", leave=False):
        x, labels = x.to(device), labels.to(device)

        fused_feat, proj, logits = model(x, return_features=True)

        loss, loss_supcon, loss_ce = criterion(fused_feat, logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_supcon += loss_supcon.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_supcon / n


def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            _, logits = model(x)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    DATA_PATH = "processed_raw_dataset.pt"
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    KNOWN_RATIO = 0.7
    SAVE_DIR = "checkpoints_supcon"

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

   
    model = OpenRFNet(num_classes=len(known_classes)).to(device)

    criterion = CombinedLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, ce_loss, supcon_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = validate(model, test_loader, device)

        print(f"Train Loss: {train_loss:.4f} | CE: {ce_loss:.4f} | SupCon: {supcon_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

   
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print("Saved new best model.")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

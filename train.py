import os
import json
import random
import pickle
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_hf_tokens():
    # Make sure a bad HF token never breaks public downloads
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME", "HF_API_TOKEN", "HF_AUTH_TOKEN"):
        os.environ.pop(k, None)
    try:
        from huggingface_hub import logout as hf_logout
        hf_logout()
    except Exception:
        pass


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Dataset
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(map(str, texts))
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# IO
# ============================================================
def load_jsonl_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


# ============================================================
# Prep helpers
# ============================================================
def compute_class_weights(y_train, n_classes, device):
    counts = Counter(y_train)
    freq = torch.tensor([counts.get(i, 0) for i in range(n_classes)], dtype=torch.float)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.mean()
    return weights.to(device)


def make_weighted_sampler(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes)
    inv = 1.0 / (counts + 1e-6)
    sample_weights = np.array([inv[y] for y in y_train], dtype=np.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def classification_report_safe(true_labels, predictions, label_encoder):
    all_label_ids = list(range(len(label_encoder.classes_)))
    target_names = [str(c) for c in label_encoder.classes_]
    return classification_report(
        true_labels, predictions,
        labels=all_label_ids,
        target_names=target_names,
        zero_division=0,
        digits=4
    )


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn=None):
    model.eval()
    total_loss = 0.0
    predictions, true_labels = [], []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=1)

        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

        if loss_fn is not None:
            total_loss += loss_fn(logits, labels).item()

    avg_loss = total_loss / max(1, len(dataloader)) if loss_fn is not None else None
    acc = accuracy_score(true_labels, predictions)
    return avg_loss, acc, predictions, true_labels


def freeze_encoder_roberta(m, freeze=True):
    # For XLM-R, encoder is "roberta"
    for p in m.roberta.parameters():
        p.requires_grad = not (freeze)


def build_optim_sched(model, lr, weight_decay, total_steps):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler


# ============================================================
# Main
# ============================================================
def main():
    # ------------ Config ------------
    set_seed(42)
    clear_hf_tokens()

    JSONL_FILE_PATH = "/home/akylbek_maxutov/openai_api/test_files/ministry_final_dataset.jsonl"
    TEXT_COL = "text"
    LABEL_COL = "label"

    MODEL_NAME = "xlm-roberta-base"   # Good for Russian
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 12
    FREEZE_EPOCHS = 2
    LR_FROZEN = 5e-5
    LR_UNFROZEN = 2e-5
    WEIGHT_DECAY = 0.01
    PATIENCE = 3
    N_SPLITS = 5
    LOG_PATH = "logs.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Reset logs
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"[{now()}] Start run\n")

    # ------------ Data ------------
    df = load_jsonl_data(JSONL_FILE_PATH)
    texts = df[TEXT_COL].astype(str).tolist()
    labels_raw = df[LABEL_COL].tolist()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_raw)
    n_classes = len(label_encoder.classes_)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] #samples={len(df)} | #classes={n_classes}\n")

    # True held-out test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] Split sizes: train+val={len(X_temp)}, test={len(X_test)}\n")

    # ------------ Tokenizer ------------
    print("\nChecking tokenizer availability…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loaded OK!")

    # ------------ CV on train+val ------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_temp, y_temp), start=1):
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] ===== Fold {fold_idx}/{N_SPLITS} =====\n")

        X_tr = [X_temp[i] for i in tr_idx]
        y_tr = [y_temp[i] for i in tr_idx]
        X_va = [X_temp[i] for i in va_idx]
        y_va = [y_temp[i] for i in va_idx]

        # Datasets + loaders
        train_ds = TextDataset(X_tr, y_tr, tokenizer, max_length=MAX_LENGTH)
        val_ds   = TextDataset(X_va, y_va, tokenizer, max_length=MAX_LENGTH)

        train_sampler = make_weighted_sampler(y_tr, n_classes)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Model/loss
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=n_classes
        ).to(device)

        class_weights = compute_class_weights(y_tr, n_classes, device)
        loss_frozen = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
        loss_unfrozen = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)

        # Phase 1: freeze encoder
        freeze_encoder_roberta(model, freeze=True)
        total_steps_frozen = len(train_loader) * max(1, FREEZE_EPOCHS)
        optimizer, scheduler = build_optim_sched(model, LR_FROZEN, WEIGHT_DECAY, total_steps_frozen)

        best_val_loss = float("inf")
        best_state = None
        patience_left = PATIENCE

        if FREEZE_EPOCHS > 0:
            for ep in range(1, FREEZE_EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_frozen)
                val_loss, val_acc, _, _ = evaluate(model, val_loader, device, loss_frozen)

                line = f"[{now()}] Fold {fold_idx} | phase=frozen | epoch={ep}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}\n"
                print(line.strip())
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = PATIENCE
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        with open(LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(f"[{now()}] Early stopping during frozen phase.\n")
                        break

        # Phase 2: unfreeze encoder
        freeze_encoder_roberta(model, freeze=False)
        if patience_left > 0:
            total_steps_unfrozen = len(train_loader) * (EPOCHS - FREEZE_EPOCHS)
            optimizer, scheduler = build_optim_sched(model, LR_UNFROZEN, WEIGHT_DECAY, total_steps_unfrozen)

            for ep in range(FREEZE_EPOCHS + 1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_unfrozen)
                val_loss, val_acc, _, _ = evaluate(model, val_loader, device, loss_unfrozen)

                line = f"[{now()}] Fold {fold_idx} | phase=unfrozen | epoch={ep}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}\n"
                print(line.strip())
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = PATIENCE
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        with open(LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(f"[{now()}] Early stopping.\n")
                        break

        # Restore best
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Fold metrics (validation set)
        _, val_acc, preds, trues = evaluate(model, val_loader, device, loss_unfrozen)
        fold_metrics.append(val_acc)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{now()}] Fold {fold_idx} final val_acc={val_acc:.4f}\n")

    # CV summary
    cv_mean = float(np.mean(fold_metrics)) if fold_metrics else 0.0
    cv_std = float(np.std(fold_metrics)) if fold_metrics else 0.0
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CV val_acc mean={cv_mean:.4f} std={cv_std:.4f}\n")
    print(f"\nCV val_acc mean={cv_mean:.4f} ± {cv_std:.4f}")

    # ------------ Final train on all train+val, evaluate on held-out test ------------
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] ===== Final model on train+val; evaluate on held-out test =====\n")

    # Make an internal small val for early stopping while training final model
    X_tr_full, X_val_small, y_tr_full, y_val_small = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )

    train_full_ds = TextDataset(X_tr_full, y_tr_full, tokenizer, max_length=MAX_LENGTH)
    val_small_ds  = TextDataset(X_val_small, y_val_small, tokenizer, max_length=MAX_LENGTH)
    test_ds       = TextDataset(X_test, y_test, tokenizer, max_length=MAX_LENGTH)

    train_full_sampler = make_weighted_sampler(y_tr_full, n_classes)
    train_full_loader = DataLoader(train_full_ds, batch_size=BATCH_SIZE, sampler=train_full_sampler)
    val_small_loader  = DataLoader(val_small_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader       = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes
    ).to(device)

    class_weights_full = compute_class_weights(y_tr_full, n_classes, device)
    loss_frozen_full = torch.nn.CrossEntropyLoss(weight=class_weights_full, label_smoothing=0.1).to(device)
    loss_unfrozen_full = torch.nn.CrossEntropyLoss(weight=class_weights_full, label_smoothing=0.1).to(device)

    # Train final with freeze→unfreeze + early stopping
    freeze_encoder_roberta(final_model, freeze=True)
    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    total_steps_frozen = len(train_full_loader) * max(1, FREEZE_EPOCHS)
    optimizer, scheduler = build_optim_sched(final_model, LR_FROZEN, WEIGHT_DECAY, total_steps_frozen)

    if FREEZE_EPOCHS > 0:
        for ep in range(1, FREEZE_EPOCHS + 1):
            train_loss = train_one_epoch(final_model, train_full_loader, optimizer, scheduler, device, loss_frozen_full)
            val_loss, val_acc, _, _ = evaluate(final_model, val_small_loader, device, loss_frozen_full)
            line = f"[{now()}] FINAL | phase=frozen | epoch={ep}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}\n"
            print(line.strip())
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
                patience_left = PATIENCE
            else:
                patience_left -= 1
                if patience_left == 0:
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(f"[{now()}] Early stopping in final (frozen phase).\n")
                    break

    freeze_encoder_roberta(final_model, freeze=False)
    if patience_left > 0:
        total_steps_unfrozen = len(train_full_loader) * (EPOCHS - FREEZE_EPOCHS)
        optimizer, scheduler = build_optim_sched(final_model, LR_UNFROZEN, WEIGHT_DECAY, total_steps_unfrozen)

        for ep in range(FREEZE_EPOCHS + 1, EPOCHS + 1):
            train_loss = train_one_epoch(final_model, train_full_loader, optimizer, scheduler, device, loss_unfrozen_full)
            val_loss, val_acc, _, _ = evaluate(final_model, val_small_loader, device, loss_unfrozen_full)
            line = f"[{now()}] FINAL | phase=unfrozen | epoch={ep}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}\n"
            print(line.strip())
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
                patience_left = PATIENCE
            else:
                patience_left -= 1
                if patience_left == 0:
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(f"[{now()}] Early stopping in final.\n")
                    break

    if best_state is not None:
        final_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{now()}] Loaded best final model by val loss.\n")

    # Test evaluation
    test_loss, test_acc, preds, trues = evaluate(final_model, test_loader, device, loss_unfrozen_full)
    rep = classification_report_safe(trues, preds, label_encoder)
    cm = confusion_matrix(trues, preds, labels=list(range(n_classes)))

    print("\n===== Held-out Test Results =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\nClassification Report:\n", rep)
    print("Confusion matrix shape:", cm.shape)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] TEST: loss={test_loss:.4f} acc={test_acc:.4f}\n")
        f.write(rep + "\n")

    # Save artifacts
    save_dir = "./xlmr_multiclass_ministry_final"
    os.makedirs(save_dir, exist_ok=True)
    final_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] Saved model/tokenizer/label_encoder to {save_dir}\n")
        f.write(f"[{now()}] Done.\n")


if __name__ == "__main__":
    main()

# model/train.py
# QED AI — Training Loop

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import time
from tqdm import tqdm

# Import our modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import QEDAI, QEDConfig
from model.tokenizer import QEDTokenizer
from data.prepare_data import load_data, prepare_for_training


def train():
    print("=" * 55)
    print("  QED AI — Training")
    print("=" * 55)

    # ── Device setup ───────────────────────────────────
    if torch.cuda.is_available():
        # Use all available GPUs
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"✅ Using {n_gpus} GPU(s)")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} ({mem:.0f}GB)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU — training will be slow!")

    # ── Hyperparameters ────────────────────────────────
    EPOCHS = 10
    BATCH_SIZE = 64  # per GPU
    LR = 3e-4
    MAX_LEN = 512
    SAVE_EVERY = 1  # save every N epochs
    LOG_EVERY = 100  # log every N steps

    # ── Load data ──────────────────────────────────────
    print("\n📦 Loading data...")
    data = load_data("data/lean4_tactics.json")

    # ── Build tokenizer ────────────────────────────────
    print("\n🔤 Building tokenizer...")
    tok_path = "saved_model/tokenizer.json"

    if os.path.exists(tok_path):
        tokenizer = QEDTokenizer.load("saved_model")
    else:
        tokenizer = QEDTokenizer()
        texts = [f"STATE: {d['state_before']} TACTIC: {d['tactic']}" for d in data]
        tokenizer.build(texts, target_vocab_size=32000)
        tokenizer.save("saved_model")

    # ── Build dataset ──────────────────────────────────
    print("\n📚 Preparing dataset...")
    dataset = prepare_for_training(data, tokenizer, MAX_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    print(f"✅ {len(dataset):,} examples, {len(dataloader):,} batches")

    # ── Build model ────────────────────────────────────
    print("\n🧠 Building model...")
    cfg = QEDConfig()
    model = QEDAI(cfg)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"✅ Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # ── Optimizer ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    # ── Training loop ──────────────────────────────────
    print("\n🚀 Training started!")
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   LR:         {LR}")
    print("-" * 55)

    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        for step, (inputs, targets) in enumerate(progress):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)  # (B, T, vocab)
            B, T, V = logits.shape

            # Calculate loss
            loss = loss_fn(logits.reshape(B * T, V), targets.reshape(B * T))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

            # Log progress
            if step % LOG_EVERY == 0:
                avg = epoch_loss / (step + 1)
                progress.set_postfix(
                    {"loss": f"{avg:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
                )

        scheduler.step()

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - train_start

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Loss:      {avg_loss:.4f}")
        print(f"  Time:      {epoch_time / 60:.1f} mins")
        print(f"  Total:     {total_time / 3600:.1f} hours")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            save_path = f"saved_model"
            os.makedirs(save_path, exist_ok=True)

            # Unwrap DataParallel if needed
            m = model.module if hasattr(model, "module") else model
            m.save(save_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                m.save("saved_model/best")
                print(f"  ✅ Best model saved! Loss: {best_loss:.4f}")

    print("\n" + "=" * 55)
    print(f"  Training complete! ∎")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Total time: {(time.time() - train_start) / 3600:.1f} hours")
    print("=" * 55)


if __name__ == "__main__":
    train()

"""
Knowledge-Distillation training for a lightweight Islamophobia detector.

Architecture (per the implementation plan):
  Teacher : distilbert-base-uncased  (66.9M params) — provides soft targets
  Student : bert-tiny                (4.4M params)  — lightweight deploy target
  Tokenization strictly shared between teacher and student (plan guidance on
  consistent sub-word vocab), so we freeze the teacher with the SAME
  distilbert tokenizer used at inference time.

Loss (plan Section: Mathematical Formulation of Distillation):
  L = alpha * CE(y_true, student) + (1-alpha) * T^2 * KL(teacher_soft, student_soft)
with temperature T=5.0 and alpha=0.5 as suggested by the plan.
"""
import argparse
import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

TEACHER_NAME = "distilbert-base-uncased"
STUDENT_NAME = "prajjwal1/bert-tiny"
MAX_LEN = 64
TEMP = 5.0
ALPHA = 0.5
BATCH_SIZE = 16


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = [t for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]


def collate(tokenizer):
    def fn(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        return enc["input_ids"], enc["attention_mask"], labels
    return fn


class DistillationTrainer:
    """Implements the plan's `HateSpeechDistillationTrainer.compute_loss`:
    hard CE + temperature-scaled KL distillation, alpha-weighted."""

    def __init__(self, teacher, student, temperature=TEMP, alpha=ALPHA):
        self.teacher = teacher
        self.teacher.eval()  # frozen
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def loss(self, input_ids, attention_mask, labels):
        # Teacher soft targets (no grad)
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.detach()

        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask).logits

        student_loss = F.cross_entropy(s_out, labels)

        student_soft = F.log_softmax(s_out / self.temperature, dim=-1)
        teacher_soft = F.softmax(t_out / self.temperature, dim=-1)
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        distillation_loss = kl * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss
        return loss, student_loss.item(), distillation_loss.item()


def evaluate(model, dataloader, device):
    correct = total = 0
    for ids, mask, labels in dataloader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        preds = logits.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # ---- data -------------------------------------------------------------
    rows = list(csv.DictReader(open("data/corpus.csv", encoding="utf-8")))
    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]
    random.shuffle(rows)
    split = int(len(rows) * 0.85)
    tr, va = rows[:split], rows[split:]
    print(f"train={len(tr)} val={len(va)}")

    # ---- models (shared tokenizer) ---------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_NAME, num_labels=2, torch_dtype=torch.float32
    ).to(device)

    # Warm-start the teacher on the task in isolation (small + fast) so its
    # soft targets are meaningful rather than a pretrained noise floor.
    teacher_warm = DistillationTrainer(
        teacher, teacher, temperature=1.0, alpha=1.0
    )
    teacher.train()
    tdl = DataLoader(
        TextDataset([r["text"] for r in tr], [int(r["label"]) for r in tr]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate(tokenizer),
    )
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=2e-5)
    for epoch in range(4):
        for ids, mask, labels in tdl:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt_t.zero_grad()
            loss, _, _ = teacher_warm.loss(ids, mask, labels)
            loss.backward()
            opt_t.step()
    teacher.eval()
    print(f"teacher warm-start acc: {evaluate(teacher, DataLoader(TextDataset([r['text'] for r in va], [int(r['label']) for r in va]), batch_size=BATCH_SIZE, collate_fn=collate(tokenizer)), device):.3f}")

    # ---- student ----------------------------------------------------------
    # prajjwal1/bert-tiny's config on the Hub has no `model_type`, so AutoConfig
    # rejects it. Reconstruct the config explicitly with model_type="bert".
    student_cfg = BertConfig(
        model_type="bert",
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    )
    student_cfg.num_labels = 2
    student = AutoModelForSequenceClassification.from_config(student_cfg)

    train_dl = DataLoader(
        TextDataset([r["text"] for r in tr], [int(r["label"]) for r in tr]),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate(tokenizer),
    )
    val_dl = DataLoader(
        TextDataset([r["text"] for r in va], [int(r["label"]) for r in va]),
        batch_size=BATCH_SIZE, collate_fn=collate(tokenizer),
    )

    trainer = DistillationTrainer(teacher, student)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        run, ce, kd = 0.0, 0.0, 0.0
        n = 0
        for ids, mask, labels in train_dl:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            loss, ce_l, kd_l = trainer.loss(ids, mask, labels)
            loss.backward()
            opt.step()
            run += loss.item(); ce += ce_l; kd += kd_l; n += 1
        student.eval()
        acc = evaluate(student, val_dl, device)
        print(f"epoch {epoch:2d} loss {run/n:5.3f} (ce {ce/n:4.3f} kd {kd/n:4.3f}) val_acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(student.state_dict(), "model/student_distilled.bin")

    print(f"best val_acc {best:.3f} -> model/student_distilled.bin")

    # Save tokenizer for inference parity with the export + tests.
    tokenizer.save_pretrained("model/tokenizer")

    # Save a full checkpoint for ONNX export.
    student.load_state_dict(torch.load("model/student_distilled.bin", weights_only=True))
    student.config.save_pretrained("model/student_config")
    torch.save(student.state_dict(), "model/student_final.bin")
    print("saved model/student_final.bin + model/tokenizer")


if __name__ == "__main__":
    main()
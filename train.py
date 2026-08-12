"""
Knowledge-Distillation training for a lightweight Islamophobia detector.

Architecture (per the implementation plan):
  Teacher : distilbert-base-uncased  (66.9M params) — provides soft targets
  Student : bert-tiny                (4.4M params)  — lightweight deploy target

Loss (plan Section: Mathematical Formulation of Distillation):
  L = alpha * CE(y_true, student) + (1-alpha) * T^2 * KL(teacher_soft, student_soft)
with temperature T=5.0 and alpha=0.5.

SPEED: teacher logits are precomputed ONCE and cached to disk; the student
loop never runs a teacher forward pass. Corpus is tokenized once up-front.
"""
import argparse
import csv
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
TEMP = 5.0
ALPHA = 0.5
MAX_LEN = 64
CACHE = "model/teacher_logits.pt"


def load_rows(path="data/corpus.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    random.seed(SEED)
    random.shuffle(rows)
    return rows


def split_rows(rows, frac=0.85):
    n = int(len(rows) * frac)
    return rows[:n], rows[n:]


def tokenize_all(tokenizer, rows):
    texts = [r["text"] for r in rows]
    enc = tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_LEN,
    )
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return enc["input_ids"], enc["attention_mask"], labels


def make_student():
    cfg = BertConfig(
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
        num_labels=2,
    )
    return AutoModelForSequenceClassification.from_config(cfg)


def evaluate(model, ids, mask, labels, batch=64):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            pred = model(
                input_ids=ids[i:i + batch], attention_mask=mask[i:i + batch]
            ).logits.argmax(-1)
            correct += (pred == labels[i:i + batch]).sum().item()
            total += labels[i:i + batch].size(0)
    return correct / max(total, 1)


def warm_start_teacher(teacher, ids, mask, labels, epochs=2, batch=32, lr=2e-5):
    """Quick supervised warm-start so soft targets are meaningful (2 epochs)."""
    ds = TensorDataset(ids, mask, labels)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(teacher.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()
    teacher.train()
    for epoch in range(epochs):
        for b_ids, b_mask, b_labels in dl:
            opt.zero_grad()
            out = teacher(input_ids=b_ids, attention_mask=b_mask).logits
            loss = loss_fct(out, b_labels)
            loss.backward()
            opt.step()
    teacher.eval()


def compute_teacher_logits(teacher, ids, mask, batch=64, cache=None):
    """Run teacher forward ONCE over all data; cache to disk for reuse."""
    if cache and os.path.exists(cache):
        t0 = time.time()
        cached = torch.load(cache, weights_only=True)
        print(f"  [cached teacher logits loaded in {time.time()-t0:.1f}s]")
        return cached
    logits = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            out = teacher(input_ids=ids[i:i + batch], attention_mask=mask[i:i + batch])
            logits.append(out.logits.detach())
    all_logits = torch.cat(logits, dim=0)
    os.makedirs(os.path.dirname(cache) if cache else ".", exist_ok=True)
    if cache:
        torch.save(all_logits, cache)
    print(f"  teacher logits computed: {all_logits.shape}")
    return all_logits


def distill(student, t_logits, ids, mask, labels,
           t_ids, t_mask, t_labels, epochs, batch=64, lr=3e-4):
    """Student learns from cached teacher logits (no teacher forward)."""
    ds = TensorDataset(t_logits, ids, mask, labels)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss()
    kld = nn.KLDivLoss(reduction="batchmean")

    best = -1.0
    for epoch in range(1, epochs + 1):
        student.train()
        run = ce = kd = 0.0
        n = 0
        for t_log, b_ids, b_mask, b_labels in dl:
            opt.zero_grad()
            s_out = student(input_ids=b_ids, attention_mask=b_mask).logits
            ce_l = loss_fct(s_out, b_labels)
            s_soft = F.log_softmax(s_out / TEMP, dim=-1)
            t_soft = F.softmax(t_log / TEMP, dim=-1)
            kd_l = kld(s_soft, t_soft) * (TEMP ** 2)
            loss = ALPHA * ce_l + (1 - ALPHA) * kd_l
            loss.backward()
            opt.step()
            run += loss.item(); ce += ce_l.item(); kd += kd_l.item(); n += 1

        acc = evaluate(student, t_ids, t_mask, t_labels)
        print(f"epoch {epoch:2d} loss {run/n:5.3f} (ce {ce/n:4.3f} kd {kd/n:4.3f}) val_acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(student.state_dict(), "model/student_distilled.bin")
    print(f"best val_acc {best:.3f} -> model/student_distilled.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t0 = time.time()
    rows = load_rows()
    tr, va = split_rows(rows)
    print(f"train={len(tr)} val={len(va)} corpus={len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    print("tokenizing corpus once...")
    ids_tr, mask_tr, lab_tr = tokenize_all(tokenizer, tr)
    ids_va, mask_va, lab_va = tokenize_all(tokenizer, va)
    print(f"tokenized in {time.time()-t0:.1f}s")

    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_NAME, num_labels=2, torch_dtype=torch.float32
    )
    warm_start_teacher(teacher, ids_tr, mask_tr, lab_tr, epochs=2)
    print(f"teacher warm-start acc: {evaluate(teacher, ids_va, mask_va, lab_va):.3f}")

    # Cache teacher logits once for the whole corpus, split into train/val.
    all_logits = compute_teacher_logits(
        teacher, torch.cat([ids_tr, ids_va]), torch.cat([mask_tr, mask_va]),
        cache=CACHE,
    )
    t_logits_tr, t_logits_va = all_logits[:len(ids_tr)], all_logits[len(ids_tr):]

    student = make_student()
    distill(student, t_logits_tr, ids_tr, mask_tr, lab_tr,
            ids_va, mask_va, lab_va, epochs=args.epochs)
    print(f"total time: {time.time()-t0:.0f}s")

    student.load_state_dict(torch.load("model/student_distilled.bin", weights_only=True))
    student.config.save_pretrained("model/student_config")
    torch.save(student.state_dict(), "model/student_final.bin")
    tokenizer.save_pretrained("model/tokenizer")
    print("saved model/student_final.bin + model/tokenizer + model/student_config")


if __name__ == "__main__":
    main()
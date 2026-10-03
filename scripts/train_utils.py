"""Shared training utilities for Condition D: QLoRA setup, the dataset wrapper, label
builders, the HAFCT loss functions, validation, logging, checkpointing."""
import os, json, csv, pickle, math, time, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def get_yn_ids(tokenizer):
    """
    Returns (yes_id, no_id).
    Research plan §5.3: must be single tokens. Verified here.
    """
    y = tokenizer.encode("Yes", add_special_tokens=False)
    n = tokenizer.encode("No",  add_special_tokens=False)
    if len(y) != 1:
        y = tokenizer.encode("yes", add_special_tokens=False)
        print(f"  Warning: 'Yes' not single token → using 'yes' id={y[0]}")
    if len(n) != 1:
        n = tokenizer.encode("no",  add_special_tokens=False)
        print(f"  Warning: 'No' not single token → using 'no' id={n[0]}")
    assert len(y)==1 and len(n)==1, "Yes/No must be single tokens"
    print(f"  yes_id={y[0]}, no_id={n[0]}")
    return y[0], n[0]


def setup_qlora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """Freezes the model, prepares it for k-bit training and attaches LoRA adapters to q_proj/v_proj."""
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

    for p in model.parameters():
        p.requires_grad = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class VQADataset(Dataset):
    """
    Loads training_data_with_cross_neg.json (or any similar JSON).
    Skips samples with missing image paths.
    """
    def __init__(self, json_path, limit=None):
        with open(json_path) as f:
            records = json.load(f)
        self.records = [r for r in records
                        if r.get("image") and os.path.exists(r["image"])]
        if limit:
            self.records = self.records[:limit]
        print(f"  Dataset: {len(self.records)} valid samples from {json_path}")

    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]


def build_labels_standard(full_input_ids, prompt_n_tokens):
    """Labels for the standard input_ids path: prompt tokens get -100, answer tokens keep their IDs."""
    labels = full_input_ids.clone()
    labels[:, :prompt_n_tokens] = -100
    return labels


def build_labels_for_merged(full_input_ids, labels_standard, v_start, v_end, merged_len):
    """Labels aligned to the merged embedding sequence: positions 0..v_end-1 get -100, the
    suffix copies labels_standard from v_end onward.
    """
    labels_merged = torch.full(
        (full_input_ids.shape[0], merged_len), -100,
        dtype=torch.long, device=full_input_ids.device
    )
    n_suffix = full_input_ids.shape[1] - v_end
    if n_suffix > 0:
        labels_merged[:, v_end:v_end + n_suffix] = labels_standard[:, v_end:]
    return labels_merged


def build_labels_pruned(labels_merged, keep_idx, v_start, v_end, seq_len):
    """
    Index labels_merged by the same all_pos used in prune_embeddings.
    Suffix labels are unchanged by pruning (they sit after visual block).
    """
    prefix_pos   = list(range(v_start))
    vis_keep_pos = (keep_idx + v_start).tolist()
    suffix_pos   = list(range(v_end, seq_len))
    all_pos = prefix_pos + vis_keep_pos + suffix_pos
    idx_t = torch.tensor(all_pos, dtype=torch.long, device=labels_merged.device)
    return labels_merged[:, idx_t]


def compute_L_CE(logits, labels):
    """Standard next-token cross-entropy with ignore_index=-100."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def compute_L_FN(logits, labels, gamma=2.0):
    """Focal cross-entropy on answer tokens: down-weights confident predictions and emphasizes hard ones."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = (shift_labels != -100)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)
    ce = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction='none')
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma * ce).mean()
    return focal


def compute_L_CIN(logits, labels, yes_id, no_id, margin=0.30):
    """Margin loss for cross-image negatives: pushes the 'No' logit above the 'Yes' logit by
    m at the last answer position.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    answer_mask = (shift_labels != -100)

    if not answer_mask.any():
        return torch.tensor(0.0, device=logits.device)

    losses = []
    for b in range(logits.shape[0]):
        pos = answer_mask[b].nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        last_pos = pos[-1].item()
        ans_logits = shift_logits[b, last_pos, :]
        no_score  = ans_logits[no_id]
        yes_score = ans_logits[yes_id]
        losses.append(F.relu(yes_score - no_score + margin))

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def compute_L_PCD(logits, labels, teacher_logit_pair, yes_id, no_id, temp=4.0):
    """KL distillation of the student's yes/no distribution toward the condition-A teacher
    logits at the last answer position.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    answer_mask  = (shift_labels != -100)

    if not answer_mask.any():
        return torch.tensor(0.0, device=logits.device)

    losses = []
    for b in range(logits.shape[0]):
        pos = answer_mask[b].nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        last_pos = pos[-1].item()
        student_yn = shift_logits[b, last_pos, [yes_id, no_id]].float() / temp
        teacher_yn = torch.tensor(teacher_logit_pair[b],
                                   device=logits.device, dtype=torch.float32) / temp
        s_prob = F.softmax(student_yn, dim=0)
        t_prob = F.softmax(teacher_yn, dim=0)
        kl = F.kl_div(s_prob.log(), t_prob, reduction='sum') * (temp ** 2)
        losses.append(kl)

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


class TrainingLogger:
    def __init__(self, loss_csv_path, val_csv_path):
        Path(loss_csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.loss_path = loss_csv_path
        self.val_path  = val_csv_path
        self.loss_rows = []
        self.val_rows  = []
        self.best_composite = -1.0
        self.best_step = 0

    def log_step(self, step, epoch, lr, total_loss, L_CE=None, L_FN=None,
                  L_CIN=None, L_PCD=None, yes_bias=None):
        row = {
            "step": step, "epoch": epoch, "lr": round(lr, 8),
            "total_loss": round(total_loss, 6),
            "L_CE":  round(L_CE,  6) if L_CE  is not None else None,
            "L_FN":  round(L_FN,  6) if L_FN  is not None else None,
            "L_CIN": round(L_CIN, 6) if L_CIN is not None else None,
            "L_PCD": round(L_PCD, 6) if L_PCD is not None else None,
            "yes_bias": round(yes_bias, 4) if yes_bias is not None else None,
            "timestamp": datetime.now().isoformat(),
        }
        self.loss_rows.append(row)
        if len(self.loss_rows) % 10 == 0:
            self._flush_loss()

    def log_val(self, step, closed_acc, yn_acc, pope_f1_neg, yes_bias):
        composite = closed_acc + pope_f1_neg
        is_best = composite > self.best_composite
        if is_best:
            self.best_composite = composite
            self.best_step = step
        row = {
            "step": step, "closed_acc": round(closed_acc, 4),
            "yn_acc": round(yn_acc, 4),
            "pope_f1_neg": round(pope_f1_neg, 4),
            "yes_bias": round(yes_bias, 4),
            "composite": round(composite, 4),
            "is_best": is_best,
        }
        self.val_rows.append(row)
        self._flush_val()
        return is_best

    def _flush_loss(self):
        if not self.loss_rows: return
        mode = "a" if os.path.exists(self.loss_path) else "w"
        with open(self.loss_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.loss_rows[0].keys()))
            if mode == "w": w.writeheader()
            w.writerows(self.loss_rows)
        self.loss_rows = []

    def _flush_val(self):
        if not self.val_rows: return
        mode = "a" if os.path.exists(self.val_path) else "w"
        with open(self.val_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.val_rows[0].keys()))
            if mode == "w": w.writeheader()
            w.writerows(self.val_rows)
        self.val_rows = []

    def flush(self):
        self._flush_loss()
        self._flush_val()


import re
def _parse_yn(t):
    t = t.strip().lower()
    if re.search(r"\byes\b", t): return 1
    if re.search(r"\bno\b",  t): return 0
    return -1

def _norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower().strip()).strip()

def run_validation(model, processor, val_records, device, max_samples=200):
    """
    Evaluate on up to 200 samples using standard model.generate (confirmed working).
    Returns (closed_acc, yn_acc, pope_f1_neg, yes_bias_rate).
    """
    from sklearn.metrics import f1_score
    model.eval()
    preds_yn, gts_yn = [], []
    preds_all, gts_all = [], []
    yes_count = 0

    samples = val_records[:max_samples]

    with torch.no_grad():
        for rec in samples:
            if not rec.get("image") or not os.path.exists(rec["image"]):
                continue
            img = Image.open(rec["image"]).convert("RGB")
            prompt = f"USER: <image>\n{rec['question']} ASSISTANT:"
            inputs = processor(text=prompt, images=img,
                               return_tensors="pt").to(device)
            out = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            n_in = inputs["input_ids"].shape[1]
            text = processor.tokenizer.decode(
                out[0][n_in:], skip_special_tokens=True).strip()

            gt  = rec.get("answer", "")
            p   = _parse_yn(text)
            g   = _parse_yn(gt)
            preds_all.append(text)
            gts_all.append(gt)
            if p != -1 and g != -1:
                preds_yn.append(p); gts_yn.append(g)
            if p == 1: yes_count += 1

    n = len(preds_all)
    if n == 0: return 0.0, 0.0, 0.0, 1.0

    yn_ok = sum(p==g for p,g in zip(preds_yn, gts_yn))
    yn_acc = yn_ok / max(len(preds_yn), 1)

    other = [(p,g) for p,g in zip(preds_all, gts_all)
             if _parse_yn(g) == -1]
    other_ok = sum(_norm(p)==_norm(g) for p,g in other)
    n_closed = len(preds_yn) + len(other)
    closed_acc = (yn_ok + other_ok) / max(n_closed, 1)

    pope_f1_neg = 0.0
    if preds_yn:
        y_true = np.array(gts_yn)
        y_pred = np.array(preds_yn)
        tp = sum(1 for p,g in zip(y_pred,y_true) if g==0 and p==0)
        fp = sum(1 for p,g in zip(y_pred,y_true) if g==1 and p==0)
        fn = sum(1 for p,g in zip(y_pred,y_true) if g==0 and p==1)
        pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1)
        pope_f1_neg = 2*pr*rc/max(pr+rc,1e-8)

    yes_bias = yes_count / n
    model.train()
    return closed_acc, yn_acc, pope_f1_neg, yes_bias


def save_checkpoint(model, optimizer, scheduler, step, out_dir, tag="best"):
    path = Path(out_dir) / tag
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, str(path / "trainer_state.pt"))
    print(f"  Saved {tag} checkpoint at step {step} → {path}")


def load_checkpoint(model, optimizer, scheduler, ckpt_dir):
    state_file = Path(ckpt_dir) / "trainer_state.pt"
    if not state_file.exists():
        return 0
    from peft import PeftModel
    state = torch.load(str(state_file), map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    print(f"  Resumed from step {state['step']}")
    return state["step"]

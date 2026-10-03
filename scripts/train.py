"""Trains the full MedPruneVLM model (paper Condition D): CAFP pruning plus HAFCT
(L_CE + L_FN + L_CIN + L_PCD). Requires data/condition_A_logits_train.pt produced
by scripts/evaluate_baseline.py.
"""
import os, sys, json, time, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, ".")

from train_utils import (
    get_yn_ids, setup_qlora, VQADataset,
    build_labels_standard, build_labels_for_merged, build_labels_pruned,
    compute_L_CE, compute_L_FN, compute_L_CIN, compute_L_PCD,
    TrainingLogger, run_validation, save_checkpoint, load_checkpoint,
)
from cafp import (
    get_merged_embeddings, run_observation_layers,
    compute_cafp_flow, get_anatomy_score,
    determine_keep_indices, prune_embeddings,
    load_anatomy_cache, CAFP_K, CAFP_R, CAFP_ALPHA,
)

CONDITION      = "D"
TRAIN_JSON     = "data/training_data_with_cross_neg.json"
VAL_JSON       = "data/vqa_rad_test.json"
POPE_JSON      = "data/pope_med_eval.json"
LOGITS_FILE    = "data/condition_A_logits_train.pt"
CKPT_DIR       = "checkpoints/condition_D"
LOG_LOSS_CSV   = "logs/training_D_loss.csv"
LOG_VAL_CSV    = "logs/training_D_validation.csv"

LR          = 2e-4
GRAD_ACCUM  = 4
MAX_EPOCHS  = 1
LOG_EVERY   = 50
SAVE_EVERY  = 500
VAL_EVERY   = 500
VAL_SAMPLES = 200
LAMBDA_FN   = 0.30
LAMBDA_CIN  = 0.20
LAMBDA_PCD  = 0.10
SEED        = 42

random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

print(f"=== Condition D — Full HAFCT ===")
print(f"Loss: L_CE + {LAMBDA_FN}*L_FN + {LAMBDA_CIN}*L_CIN + {LAMBDA_PCD}*L_PCD")

assert os.path.exists(LOGITS_FILE), \
    f"MISSING: {LOGITS_FILE}\nRun: python scripts/evaluate_baseline.py first"
teacher_logits = torch.load(LOGITS_FILE, map_location="cpu", weights_only=True)
print(f"Loaded teacher logits: {len(teacher_logits)} samples")

from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16)
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", quantization_config=bnb,
    device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager")
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

yes_id, no_id = get_yn_ids(processor.tokenizer)
anatomy_cache = load_anatomy_cache(["vqa_rad"])

model = setup_qlora(model)
model.gradient_checkpointing_disable()
dataset = VQADataset(TRAIN_JSON)
val_records = []
if os.path.exists(VAL_JSON):
    with open(VAL_JSON) as f: val_records = json.load(f)
if os.path.exists(POPE_JSON):
    with open(POPE_JSON) as f:
        val_records = val_records[:100] + json.load(f)[:100]

optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
total_steps = (len(dataset) * MAX_EPOCHS) // GRAD_ACCUM
scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR*0.1)
logger      = TrainingLogger(LOG_LOSS_CSV, LOG_VAL_CSV)

resume_step = 0
latest_ckpt = Path(CKPT_DIR) / "latest"
if latest_ckpt.exists():
    resume_step = load_checkpoint(model, optimizer, scheduler, latest_ckpt)

print("\n=== Diagnostic: one full HAFCT step ===")
_rec = next(r for r in dataset.records if r.get("id") in teacher_logits)
_img = Image.open(_rec["image"]).convert("RGB")
_prompt = f"USER: <image>\n{_rec['question']} ASSISTANT:"
_full_inputs   = processor(text=_prompt+f" {_rec['answer']}", images=_img, return_tensors="pt").to(device)
_prompt_inputs = processor(text=_prompt, images=_img, return_tensors="pt")
_n_prompt = _prompt_inputs["input_ids"].shape[1]
_labels_std = build_labels_standard(_full_inputs["input_ids"], _n_prompt)

print(f"  input_ids shape: {_full_inputs['input_ids'].shape}")
print(f"  n_prompt: {_n_prompt}, answer tokens: {_full_inputs['input_ids'].shape[1]-_n_prompt}")
print(f"  answer ids: {_full_inputs['input_ids'][0,_n_prompt:].tolist()}")


with torch.no_grad():
    _merged, _mask, _vs, _ve = get_merged_embeddings(
        model, _full_inputs["input_ids"], _full_inputs["pixel_values"],
        _full_inputs.get("attention_mask"))
    _lm = build_labels_for_merged(_full_inputs["input_ids"], _labels_std, _vs, _ve, _merged.shape[1])
    _, _hk = run_observation_layers(model, _merged, _mask, CAFP_K)
    _flow  = compute_cafp_flow({}, _ve-_vs, _vs, CAFP_K, hidden_states=_hk); del _hk
    _smed  = get_anatomy_score(_rec.get("id"), _rec["question"], anatomy_cache, _ve-_vs)
    _kidx  = determine_keep_indices(_flow, _smed, CAFP_ALPHA, CAFP_R, _ve-_vs)
    _pe, _pm = prune_embeddings(_merged, _mask, _kidx, _vs, _ve)
    _lp    = build_labels_pruned(_lm, _kidx, _vs, _ve, _merged.shape[1])
    del _merged
_out = model(inputs_embeds=_pe, attention_mask=_pm, use_cache=False, return_dict=True)
_ce = compute_L_CE(_out.logits, _lp)
_fn = compute_L_FN(_out.logits, _lp)
_cin = compute_L_CIN(_out.logits, _lp, yes_id, no_id)
_tl  = teacher_logits.get(_rec.get("id"), [0.0, 0.0])
_pcd = compute_L_PCD(_out.logits, _lp, [_tl], yes_id, no_id)
print(f"  L_CE={_ce.item():.4f}  L_FN={_fn.item():.4f}"
      f"  L_CIN={_cin.item():.4f}  L_PCD={_pcd.item():.4f}")
del _out, _pe, _pm, _lp, _full_inputs, _prompt_inputs
torch.cuda.empty_cache()
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("=== Diagnostic passed ===\n")

model.train()
global_step  = 0
optimizer.zero_grad()
accum = dict(total=0., ce=0., fn=0., cin=0., pcd=0.)
indices = list(range(len(dataset))); random.shuffle(indices)

for epoch in range(MAX_EPOCHS):
    for i, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch+1}")):
        global_step += 1
        if global_step <= resume_step: continue

        rec = dataset.records[idx]
        if not rec.get("image") or not os.path.exists(rec["image"]): continue

        is_cross_neg = rec.get("cross_image_neg", False)
        sample_id    = rec.get("id", "")
        teacher_pair = teacher_logits.get(sample_id, [0.0, 0.0])

        try:
            img    = Image.open(rec["image"]).convert("RGB")
            prompt = f"USER: <image>\n{rec['question']} ASSISTANT:"
            answer = f" {rec['answer']}"

            full_inputs   = processor(text=prompt+answer, images=img, return_tensors="pt").to(device)
            prompt_inputs = processor(text=prompt,        images=img, return_tensors="pt")
            n_prompt      = prompt_inputs["input_ids"].shape[1]
            labels_std    = build_labels_standard(full_inputs["input_ids"], n_prompt)

            with torch.no_grad():
                merged, mask, v_start, v_end = get_merged_embeddings(
                    model, full_inputs["input_ids"], full_inputs["pixel_values"],
                    full_inputs.get("attention_mask"))
                labels_m = build_labels_for_merged(
                    full_inputs["input_ids"], labels_std, v_start, v_end, merged.shape[1])
                _, hidden_k = run_observation_layers(model, merged, mask, CAFP_K)
                flow = compute_cafp_flow({}, v_end - v_start, v_start, CAFP_K,
                                         hidden_states=hidden_k)
                del hidden_k
                s_med = get_anatomy_score(sample_id, rec["question"],
                                           anatomy_cache, v_end-v_start)
                keep_idx = determine_keep_indices(flow, s_med, CAFP_ALPHA, CAFP_R,
                                                   v_end-v_start)
                pruned_e, pruned_m = prune_embeddings(merged, mask, keep_idx, v_start, v_end)
                labels_p = build_labels_pruned(labels_m, keep_idx, v_start, v_end,
                                                merged.shape[1])
                del merged

            out = model(inputs_embeds=pruned_e, attention_mask=pruned_m,
                         use_cache=False, return_dict=True)

            L_CE_v  = compute_L_CE(out.logits, labels_p)
            L_FN_v  = compute_L_FN(out.logits, labels_p)
            L_CIN_v = compute_L_CIN(out.logits, labels_p, yes_id, no_id) \
                      if is_cross_neg else torch.tensor(0.0, device=device)
            L_PCD_v = compute_L_PCD(out.logits, labels_p, [teacher_pair], yes_id, no_id)

            total = (L_CE_v
                     + LAMBDA_FN  * L_FN_v
                     + LAMBDA_CIN * L_CIN_v
                     + LAMBDA_PCD * L_PCD_v)
            (total / GRAD_ACCUM).backward()

            accum["total"] += total.item()
            accum["ce"]    += L_CE_v.item()
            accum["fn"]    += L_FN_v.item()
            accum["cin"]   += L_CIN_v.item()
            accum["pcd"]   += L_PCD_v.item()

        except Exception as e:
            print(f"\n  Error step {global_step}: {e}")
            optimizer.zero_grad()
            for k in accum: accum[k] = 0.0
            torch.cuda.empty_cache(); continue
        finally:
            for v in ['full_inputs','prompt_inputs','out','pruned_e','pruned_m','labels_p']:
                if v in dir(): del v
            torch.cuda.empty_cache()

        if global_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if (global_step // GRAD_ACCUM) % LOG_EVERY == 0:
                lr_now = scheduler.get_last_lr()[0]
                n = LOG_EVERY
                logger.log_step(global_step, epoch+1, lr_now,
                                 accum["total"]/n,
                                 L_CE=accum["ce"]/n, L_FN=accum["fn"]/n,
                                 L_CIN=accum["cin"]/n, L_PCD=accum["pcd"]/n)
                print(f"  step={global_step}  total={accum['total']/n:.4f}"
                      f"  CE={accum['ce']/n:.4f}  FN={accum['fn']/n:.4f}"
                      f"  CIN={accum['cin']/n:.4f}  PCD={accum['pcd']/n:.4f}")
                for k in accum: accum[k] = 0.0

            if (global_step // GRAD_ACCUM) % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler, global_step,
                                 CKPT_DIR, "latest")
            if (global_step // GRAD_ACCUM) % VAL_EVERY == 0:
                ca, ya, pf, yb = run_validation(model, processor, val_records,
                                                  device, VAL_SAMPLES)
                is_best = logger.log_val(global_step, ca, ya, pf, yb)
                print(f"  [VAL] closed={ca:.4f}  pope_f1={pf:.4f}  yes_bias={yb:.4f}")
                if is_best:
                    save_checkpoint(model, optimizer, scheduler, global_step,
                                     CKPT_DIR, "best")

logger.flush()
save_checkpoint(model, optimizer, scheduler, global_step, CKPT_DIR, "final")
print(f"\nCondition D complete. Best step: {logger.best_step}")

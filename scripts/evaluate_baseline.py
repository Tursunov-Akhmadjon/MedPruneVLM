"""Evaluates the zero-shot LLaVA-1.5-7B-hf baseline (paper Condition A) on all test
sets and extracts teacher yes/no logits for the L_PCD distillation loss.
"""

import os, sys, csv, json, re, time, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "condition_A"
LOGS_DIR     = PROJECT_ROOT / "logs"

CONDITION    = "A"
MODEL_ID     = "llava-hf/llava-1.5-7b-hf"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

IU_XRAY_PROMPT = (
    "You are a radiologist reviewing a chest X-ray. "
    "Provide a structured radiology report describing your findings."
)

REQUIRED_FILES = {
    "vqa_rad_test":  DATA_DIR / "vqa_rad_test.json",
    "slake_test":    DATA_DIR / "slake_test_en.json",
    "pope_med_eval": DATA_DIR / "pope_med_eval.json",
    "training_data": DATA_DIR / "training_data_with_cross_neg.json",
}
OPTIONAL_FILES = {
    "iu_xray_test":  DATA_DIR / "iu_xray_test.json",
}

def ts():    return datetime.now().strftime("%H:%M:%S")
def log(m):  print(f"[{ts()}] [INFO]  {m}", flush=True)
def warn(m): print(f"[{ts()}] [WARN]  {m}", flush=True)
def err(m):  print(f"[{ts()}] [ERROR] {m}", flush=True)
def section(t):
    bar = "═" * 70
    print(f"\n{bar}\n  {t}\n{bar}", flush=True)

def check_prerequisites():
    section("PREREQUISITE CHECK")
    ok = True
    for key, p in REQUIRED_FILES.items():
        if not p.exists():
            err(f"  ✗  MISSING: {p.name}")
            err(f"     Run the script that creates '{key}' first.")
            ok = False
        else:
            n = len(_load_json(p))
            log(f"  ✓  {p.name:46s}  {n:,} samples")
    for key, p in OPTIONAL_FILES.items():
        if p.exists():
            n = len(_load_json(p))
            sym = "✓" if n > 0 else "○ (empty)"
            log(f"  {sym}  {p.name:46s}  {n:,} samples  [optional]")
        else:
            log(f"  ○  {p.name:46s}  not found  [optional — will skip]")
    if not ok:
        sys.exit(1)
    log("All required prerequisites met.")

def load_model():
    section(f"LOADING MODEL: {MODEL_ID}  (4-bit NF4)")
    from transformers import (
        LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer

    y_enc = tok.encode("Yes", add_special_tokens=False)
    n_enc = tok.encode("No",  add_special_tokens=False)
    if len(y_enc) != 1:
        y_enc = tok.encode("yes", add_special_tokens=False)
        warn(f"  'Yes' is not a single token — falling back to 'yes' (id={y_enc[0]})")
    if len(n_enc) != 1:
        n_enc = tok.encode("no",  add_special_tokens=False)
        warn(f"  'No'  is not a single token — falling back to 'no'  (id={n_enc[0]})")

    yes_id, no_id = y_enc[0], n_enc[0]
    vram = torch.cuda.memory_allocated() / 1e9
    log(f"  Model loaded.  VRAM: {vram:.2f} GB  |  Yes-id: {yes_id}  No-id: {no_id}")
    return model, processor, yes_id, no_id

def _fmt_prompt(question: str) -> str:
    """LLaVA-1.5 instruction format. Do not alter this string."""
    return f"USER: <image>\n{question}\nASSISTANT:"

def _load_image(rel_path: str) -> PILImage.Image:
    return PILImage.open(PROJECT_ROOT / rel_path).convert("RGB")

@torch.no_grad()
def run_generate(model, processor, rec: dict, max_new: int) -> str:
    """
    Generate a response for one record.
    Returns the model's decoded response (prompt prefix stripped).
    """
    image   = _load_image(rec["image"])
    prompt  = _fmt_prompt(rec["question"])
    inputs  = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    n_in    = inputs["input_ids"].shape[-1]
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new,
        do_sample=False, temperature=1.0,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.tokenizer.decode(
        out_ids[0][n_in:], skip_special_tokens=True
    ).strip()


@torch.no_grad()
def run_logits(model, processor, rec: dict, yes_id: int, no_id: int) -> list:
    """Forward pass only; returns [logit_yes, logit_no] at the next-token position."""
    image  = _load_image(rec["image"])
    prompt = _fmt_prompt(rec["question"])
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    out    = model(**inputs, return_dict=True)
    last   = out.logits[0, -1]
    return [last[yes_id].item(), last[no_id].item()]

def _parse_yn(response: str):
    """
    Parse a yes/no response to 1 (yes), 0 (no) or None (unparseable).
    """
    r = response.strip().lower()
    if r.startswith("yes"): return 1
    if r.startswith("no"):  return 0
    words = r.split()
    if words and words[0] in ("yes", "yeah", "true"):  return 1
    if words and words[0] in ("no", "nope", "false"):  return 0
    return None

def _norm(s: str) -> str:
    """Lowercase + strip punctuation for closed-ended exact match."""
    return re.sub(r"[^\w\s]", "", str(s).strip().lower()).strip()

def compute_vqa_metrics(rows: list) -> dict:
    """Returns closed-ended accuracy and yes/no accuracy; non-yes/no answers use exact match."""
    closed = [r for r in rows if r.get("question_type") != "report"]

    def is_yesno(qtype):
        return str(qtype).lower() in {"yes_no", "yes/no", "yn"}

    yn = [r for r in closed if is_yesno(r.get("question_type"))]
    non_yn = [r for r in closed if not is_yesno(r.get("question_type"))]

    n_yn_correct = 0
    for r in yn:
        pred = _parse_yn(r["model_output"])
        gt   = 1 if _norm(r["answer"]).startswith("yes") else 0
        if pred is not None and pred == gt:
            n_yn_correct += 1

    n_non_yn_correct = sum(
        _norm(r["model_output"]) == _norm(r["answer"]) for r in non_yn
    )

    total_correct = n_yn_correct + n_non_yn_correct


    def token_f1(pred, gt):
        pred_tokens = set(_norm(pred).split())
        gt_tokens = set(_norm(gt).split())

        if not pred_tokens or not gt_tokens:
            return 0.0

        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
    open_scores = [token_f1(r["model_output"], r["answer"]) for r in non_yn]

    return {
        "n_total":    len(rows),
        "n_closed":   len(closed),
        "n_yn":       len(yn),
        "closed_acc": round(total_correct      / max(len(closed), 1), 4),
        "yn_acc":     round(n_yn_correct       / max(len(yn),     1), 4),
        "open_token_f1": round(np.mean(open_scores), 4)
    }

def compute_pope_metrics(rows: list) -> dict:
    """
    Full POPE-Med metric set; the negative class (label=0) is the anti-hallucination class.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score
    )
    y_true, y_pred, n_unparse = [], [], 0

    for r in rows:
        gt = 1 if str(r["answer"]).strip().lower().startswith("yes") else 0
        pr = _parse_yn(r["model_output"])
        if pr is None:
            n_unparse += 1
            pr = 1
        y_true.append(gt); y_pred.append(pr)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cross = [r for r in rows if r.get("cross_image_neg")]
    if cross:
        ct = np.array([1 if str(r["answer"]).lower().startswith("yes") else 0 for r in cross])
        cp = np.array([_parse_yn(r["model_output"]) if _parse_yn(r["model_output"]) is not None else 1 for r in cross])
        cross_f1 = float(f1_score(ct, cp, pos_label=0, zero_division=0))
    else:
        cross_f1 = 0.0

    return {
        "n_total":          len(rows),
        "n_unparseable":    n_unparse,
        "unparse_rate":     round(n_unparse / max(len(rows), 1), 4),
        "overall_acc":      round(float(accuracy_score(y_true, y_pred)), 4),
        "overall_f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_neg":    round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_neg":       round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_neg":           round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "yes_bias_rate":    round(float(np.mean(y_pred)), 4),
        "cross_img_f1_neg": round(cross_f1, 4),
        "n_cross_img":      len(cross),
        "n_original":       len(rows) - len(cross),
    }

def compute_iu_xray_metrics(rows: list) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    bleu_scores = []
    rouge_scores = []

    predictions = []
    references = []

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    smooth_fn = SmoothingFunction().method1

    for r in rows:

        gt = str(r["answer"])
        pred = str(r["model_output"])

        predictions.append(pred)
        references.append(gt)

        bleu1 = sentence_bleu(
            [gt.split()],
            pred.split(),
            weights=(1.0, 0, 0, 0),
            smoothing_function=smooth_fn
        )

        bleu_scores.append(bleu1)

        rouge_f1 = scorer.score(gt, pred)["rougeL"].fmeasure

        rouge_scores.append(rouge_f1)

    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        verbose=False
    )

    return {
        "open_bleu1": round(float(np.mean(bleu_scores)), 4),
        "open_rougeL_f1": round(float(np.mean(rouge_scores)), 4),
        "open_bertscore_f1": round(float(F1.mean()), 4),
    }

_METRICS_TMPL = dict.fromkeys([
    "condition", "dataset",
    "closed_accuracy", "yes_no_accuracy", "open_token_f1",

    "open_bleu1", "open_rougeL_f1", "open_bertscore_f1",

    "pope_med_f1_neg", "pope_med_precision_neg",
    "pope_med_recall_neg", "pope_med_yes_bias", "pope_med_f1_neg_crossimg",

    "ner_entity_precision", "ner_entity_recall",

    "radgraph_f1", "radgraph_precision",

    "mean_arc", "mean_flow_entropy", "anatomy_protection_rate",

    "inference_latency_ms", "inference_latency_std",
    "peak_vram_gb", "n_samples",
])

def _load_json(path, limit=None):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d[:limit] if limit else d

def _save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _write_raw_csv(rows: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_outputs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id","dataset","image_path","question","ground_truth",
            "model_output","question_type","report_type","cross_image_neg","condition"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "sample_id":       r.get("id",""),
                "dataset":         r.get("dataset",""),
                "image_path":      r.get("image",""),
                "question":        r.get("question",""),
                "ground_truth":    r.get("answer",""),
                "model_output":    r.get("model_output",""),
                "question_type":   r.get("question_type",""),
                "report_type":     r.get("report_type",""),
                "cross_image_neg": r.get("cross_image_neg", False),
                "condition":       CONDITION,
            })

def _write_config(out_dir: Path, dataset: str, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json({
        "condition": CONDITION, "dataset": dataset,
        "model_checkpoint": MODEL_ID, "quantization": "4bit-nf4",
        "cafp_K": None, "cafp_r": None, "cafp_alpha": None,
        "hafct_lambda_fn": None, "hafct_lambda_cin": None, "hafct_lambda_pcd": None,
        "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat(),
        "n_samples": n,
    }, out_dir / "run_config.json")

def _write_metrics(updates: dict, out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    m = dict(_METRICS_TMPL)
    m.update({"condition": CONDITION, "dataset": dataset})
    m.update(updates)
    _save_json(m, out_dir / "metrics.json")

def eval_vqa_dataset(model, processor, yes_id, no_id,
                     records: list, ds_name: str, max_new: int) -> dict:
    """Run generation + VQA metrics on a standard VQA/report dataset."""
    out_dir = RESULTS_DIR / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, lats = [], []
    torch.cuda.reset_peak_memory_stats()

    log("  GPU warm-up (10 samples)…")
    for rec in records[:10]:
        try: run_generate(model, processor, rec, max_new)
        except Exception: pass
    torch.cuda.empty_cache()

    for rec in tqdm(records, desc=ds_name, ncols=80):
        t0 = time.perf_counter()
        try:
            out = run_generate(model, processor, rec, max_new)
        except Exception as e:
            warn(f"  Error on {rec.get('id','?')}: {e}")
            out = ""
        lats.append((time.perf_counter() - t0) * 1000)
        rows.append({**rec, "model_output": out})

    peak_vram = (torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0)
    lat_arr   = np.array(lats[10:])
    m         = compute_vqa_metrics(rows)

    _write_raw_csv(rows, out_dir)
    _save_json(rows, out_dir / "predictions.json")
    _write_config(out_dir, ds_name, len(rows))
    _write_metrics({
        "closed_accuracy":       m["closed_acc"],
        "yes_no_accuracy":       m["yn_acc"],
        "open_token_f1":         m["open_token_f1"],
        "inference_latency_ms":  round(float(lat_arr.mean()), 2),
        "inference_latency_std": round(float(lat_arr.std()),  2),
        "peak_vram_gb":          round(peak_vram, 3),
        "n_samples":             len(rows),
    }, out_dir, ds_name)

    log(f"  ✓ {ds_name}: closed_acc={m['closed_acc']:.3f}  "
        f"yn_acc={m['yn_acc']:.3f} open_f1={m['open_token_f1']:.3f}  latency={lat_arr.mean():.0f}ms")
    return m


def eval_pope_med(model, processor, yes_id, no_id, records: list) -> dict:
    """
    POPE-Med hallucination benchmark evaluation.
    Primary metric: f1_neg (negative-class F1 = anti-hallucination F1).
    """
    out_dir = RESULTS_DIR / "pope_med"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, lats = [], []
    torch.cuda.reset_peak_memory_stats()
    for rec in records[:10]:
        try: run_generate(model, processor, rec, 10)
        except Exception: pass

    for rec in tqdm(records, desc="POPE-Med", ncols=80):
        t0 = time.perf_counter()
        try:
            out = run_generate(model, processor, rec, 10)
        except Exception as e:
            warn(f"  Error on {rec.get('id','?')}: {e}")
            out = ""
        lats.append((time.perf_counter() - t0) * 1000)
        rows.append({**rec, "model_output": out})

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    lat_arr   = np.array(lats[10:])
    pm        = compute_pope_metrics(rows)

    _write_raw_csv(rows, out_dir)
    _save_json(rows, out_dir / "predictions.json")
    _write_config(out_dir, "pope_med", len(rows))
    _write_metrics({
        "pope_med_f1_neg":          pm["f1_neg"],
        "pope_med_precision_neg":   pm["precision_neg"],
        "pope_med_recall_neg":      pm["recall_neg"],
        "pope_med_yes_bias":        pm["yes_bias_rate"],
        "pope_med_f1_neg_crossimg": pm["cross_img_f1_neg"],
        "inference_latency_ms":     round(float(lat_arr.mean()), 2),
        "inference_latency_std":    round(float(lat_arr.std()),  2),
        "peak_vram_gb":             round(peak_vram, 3),
        "n_samples":                pm["n_total"],
    }, out_dir, "pope_med")

    print()
    print(f"  ┌───────────────────────────────────────────────────────")
    print(f"  │  POPE-Med results (Condition A anchor — record these)")
    print(f"  │")
    print(f"  │  F1 neg-class (PRIMARY):  {pm['f1_neg']:.4f}")
    print(f"  │  Precision neg:           {pm['precision_neg']:.4f}")
    print(f"  │  Recall neg:              {pm['recall_neg']:.4f}")
    print(f"  │  Overall accuracy:        {pm['overall_acc']:.4f}")
    print(f"  │  Yes-bias rate:           {pm['yes_bias_rate']:.4f}  (ideal < 0.50)")
    print(f"  │  Cross-img F1 neg:        {pm['cross_img_f1_neg']:.4f}")
    print(f"  │  Unparseable responses:   {pm['n_unparseable']}  ({pm['unparse_rate']:.1%})")
    print(f"  │  (if unparse_rate > 5%, model output format is broken)")
    print(f"  └───────────────────────────────────────────────────────")
    print()

    return pm

def eval_ui_xray_dataset(model, processor, records, max_new: int) -> dict:
    """Run generation + IU-xray metrics on a standard dataset."""

    out_dir = RESULTS_DIR / "iu_xray"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, lats = [], []
    torch.cuda.reset_peak_memory_stats()

    log("  GPU warm-up (10 samples)…")
    for rec in records[:10]:
        try: run_generate(model, processor, rec, max_new)
        except Exception: pass
    torch.cuda.empty_cache()

    for rec in tqdm(records, desc="IU-Xray", ncols=80):
        t0 = time.perf_counter()
        try:
            out = run_generate(model, processor, rec, max_new)
        except Exception as e:
            warn(f"  Error on {rec.get('id','?')}: {e}")
            out = ""
        lats.append((time.perf_counter() - t0) * 1000)
        rows.append({**rec, "model_output": out})

    peak_vram = (torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0)
    lat_arr   = np.array(lats[10:])
    um         = compute_iu_xray_metrics(rows)

    _write_raw_csv(rows, out_dir)
    _save_json(rows, out_dir / "predictions.json")
    _write_config(out_dir, "iu_xray", len(rows))
    _write_metrics({
        "open_bleu1":       um["open_bleu1"],
        "open_rougeL_f1":       um["open_rougeL_f1"],
        "open_bertscore_f1":         um["open_bertscore_f1"],
        "inference_latency_ms":  round(float(lat_arr.mean()), 2),
        "inference_latency_std": round(float(lat_arr.std()),  2),
        "peak_vram_gb":          round(peak_vram, 3),
        "n_samples":             len(rows),
    }, out_dir, "iu_xray")

    log(f"  ✓ iu_xray: BLEU1={um['open_bleu1']:.3f}  "
        f"ROUGE-L={um['open_rougeL_f1']:.3f} BERTScore={um['open_bertscore_f1']:.3f}")
    return um

def extract_training_logits(model, processor, yes_id: int, no_id: int,
                            train_records: list):
    """Extracts {sample_id: [logit_yes, logit_no]} for every training sample and saves it to
    data/condition_A_logits_train.pt, which the training conditions load for the L_PCD loss.
    """
    section("LOGIT EXTRACTION — training set for L_PCD loss")
    log(f"  {len(train_records):,} training samples")
    log(f"  Estimated: ~{len(train_records)//350:.0f} min on RTX 4060")
    log(f"  Each sample: one forward pass (no generation) — faster than eval")

    logits_dict = {}
    errors      = 0

    for rec in tqdm(train_records, desc="logit extraction", ncols=80):
        try:
            logits_dict[rec["id"]] = run_logits(model, processor, rec, yes_id, no_id)
        except Exception as e:
            errors += 1
            if errors <= 5:
                warn(f"  Error on {rec.get('id', '?')}: {e}")

    log(f"  Extracted: {len(logits_dict):,}  |  Errors: {errors}")

    out_path = DATA_DIR / "condition_A_logits_train.pt"
    torch.save(logits_dict, str(out_path))

    loaded   = torch.load(str(out_path), map_location="cpu")
    coverage = len(set(loaded.keys()) & {r["id"] for r in train_records})
    cov_pct  = coverage / max(len(train_records), 1)

    log(f"  ✓ condition_A_logits_train.pt  ({len(loaded):,} entries)")
    log(f"  Coverage: {coverage:,}/{len(train_records):,} ({cov_pct:.1%})")

    if cov_pct < 0.95:
        warn("  Coverage < 95%. Some images likely failed to load.")
        warn("  Check image paths in training_data_with_cross_neg.json.")
        warn("  Training will still work; L_PCD will be skipped on missing samples.")

    sample_ids = list(logits_dict.keys())[:3]
    log("  Sample logit pairs (should not be both ~0.0):")
    for sid in sample_ids:
        ly, ln = logits_dict[sid]
        log(f"    {sid[:40]:40s}  logit_yes={ly:+.3f}  logit_no={ln:+.3f}")

def main():
    ap = argparse.ArgumentParser(
        description="Condition A: zero-shot LLaVA-1.5-7B baseline inference")
    ap.add_argument("--skip-logits",  action="store_true",
                    help="Skip logit extraction. DO NOT use for the real run.")
    ap.add_argument("--skip-iu-xray", action="store_true",
                    help="Skip IU X-Ray eval (auto-skipped if file is empty)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap samples per dataset. For smoke testing only.")
    args = ap.parse_args()

    if args.skip_logits:
        warn("──────────────────────────────────────────────────────")
        warn("  --skip-logits is set.")
        warn("  condition_A_logits_train.pt will NOT be created.")
        warn("  You MUST re-run without --skip-logits before training.")
        warn("──────────────────────────────────────────────────────")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("vqa_rad","slake","iu_xray","pope_med"):
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    check_prerequisites()

    model, processor, yes_id, no_id = load_model()

    lim = args.limit
    vqa_test   = _load_json(REQUIRED_FILES["vqa_rad_test"],  lim)
    slake_test = _load_json(REQUIRED_FILES["slake_test"],    lim)
    pope_eval  = _load_json(REQUIRED_FILES["pope_med_eval"], lim)
    train_data = _load_json(REQUIRED_FILES["training_data"], lim)

    iu_test = []
    iu_path = OPTIONAL_FILES["iu_xray_test"]
    if not args.skip_iu_xray and iu_path.exists():
        iu_test = _load_json(iu_path, lim)

    summary = {}

    section("CONDITION A — VQA-RAD TEST")
    m = eval_vqa_dataset(model, processor, yes_id, no_id,
                         vqa_test, "vqa_rad", max_new=30)
    summary["vqa_rad"] = m
    torch.cuda.empty_cache()

    section("CONDITION A — SLAKE TEST (English only)")
    m = eval_vqa_dataset(model, processor, yes_id, no_id,
                         slake_test, "slake", max_new=30)
    summary["slake"] = m
    torch.cuda.empty_cache()

    if len(iu_test) > 0:
        section("CONDITION A — IU X-RAY TEST (report generation)")
        m = eval_ui_xray_dataset(model, processor, iu_test, max_new=256)
        summary["iu_xray"] = m
        torch.cuda.empty_cache()
    else:
        log("  IU X-Ray skipped (file empty or --skip-iu-xray set).")
        log("  Run convert_iu_xray.py first, then re-run this script for IU X-Ray eval.")

    section("CONDITION A — POPE-MED EVALUATION (primary hallucination metric)")
    pm = eval_pope_med(model, processor, yes_id, no_id, pope_eval)
    summary["pope_med"] = pm
    torch.cuda.empty_cache()

    if not args.skip_logits:
        extract_training_logits(model, processor, yes_id, no_id, train_data)
    else:
        warn("Logit extraction SKIPPED. Re-run without --skip-logits before training.")

    _save_json(summary, RESULTS_DIR / "condition_A_summary.json")

    section("CONDITION A — COMPLETE")
    print()
    print("  Dataset            closed_acc   yn_acc   n_samples")
    print("  ─────────────────────────────────────────────────")
    for ds, m in summary.items():
        if ds == "pope_med":
            continue

        if ds == "iu_xray":
            print(f"  {ds:18s} "
                f"BLEU1={m.get('bleu1',0):.3f}  "
                f"ROUGE-L={m.get('rougeL_f1',0):.3f}  "
                f"BERTScore={m.get('bertscore_f1',0):.3f}")

        else:
            print(f"  {ds:18s} "
                f"closed_acc={m.get('closed_acc',0):.3f}  "
                f"yn_acc={m.get('yn_acc',0):.3f}  "
                f"n={m.get('n_total',0):,}")
    print()
    pm_data = summary.get("pope_med", {})
    print(f"  POPE-Med F1 (neg):   {pm_data.get('f1_neg', 0):.4f}  ← Condition A anchor")
    print(f"  Yes-bias rate:       {pm_data.get('yes_bias_rate', 0):.4f}  ← expect to fall under HAFCT")
    print()
    logits_path = DATA_DIR / "condition_A_logits_train.pt"
    if logits_path.exists():
        log(f"Logits file: {logits_path.relative_to(PROJECT_ROOT)}")
        log("condition_A_logits_train.pt exists — training runs are unblocked.")
    else:
        warn("condition_A_logits_train.pt MISSING — re-run without --skip-logits.")
    print()
    log("Results: results/condition_A/")
    log("Next:    uv run scripts/build_anatomy_maps.py   (SAM-Med2D batch job, ~4 hrs)")


if __name__ == "__main__":
    main()

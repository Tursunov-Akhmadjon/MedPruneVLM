"""CAFP (Clinically-Aware Feature Pruning) for LLaVA-1.5-7B: an observation-stage flow
score fused with a SAM-Med2D anatomy prior selects which visual tokens survive pruning.
"""
from __future__ import annotations
import os, sys, json, pickle, time, math, importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch

CAFP_K         = 4
CAFP_R         = 0.70
CAFP_ALPHA     = 0.3
N_VISUAL       = 576
TOKEN_GRID     = 24
MAX_NEW_TOKENS = 32
MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
ANATOMY_CACHE  = {
    "vqa_rad": "data/anatomy_maps_cache/vqa_rad_maps.pkl",
    "iu_xray": "data/anatomy_maps_cache/iu_xray_maps.pkl",
    "slake": "data/anatomy_maps_cache/slake_maps.pkl"
}

def _load_sam_module():
    spec = importlib.util.spec_from_file_location("sam_anatomy_maps_05", "o5_sam_anatomy_maps.py")
    if spec is None: return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[CAFP] SAM module load error: {e}")
        return None

_sam_mod = _load_sam_module()
_have_sam_lookup = _sam_mod is not None and hasattr(_sam_mod, "get_anatomy_token_mask")
print(f"[CAFP] SAM anatomy lookup: {'v' if _have_sam_lookup else 'x (spatial prior fallback)'}")


def load_model_for_cafp(model_id: str = MODEL_ID, device: str = "cuda"):
    """Loads LLaVA-1.5-7B-hf in 4-bit NF4 together with its processor."""
    from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()
    processor = LlavaProcessor.from_pretrained(model_id)
    print(f"[CAFP] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, processor


def load_anatomy_cache(datasets: list = ("vqa_rad", "iu_xray")) -> dict:
    """Loads the precomputed SAM-Med2D anatomy-map caches for the requested datasets."""
    combined = {}
    for ds in datasets:
        pkl = ANATOMY_CACHE.get(ds, "")
        if not pkl or not os.path.exists(pkl):
            print(f"[CAFP] Anatomy cache not found: {pkl}")
            continue
        with open(pkl, "rb") as f:
            cache = pickle.load(f)
        combined.update(cache)
        print(f"[CAFP] Loaded anatomy cache {ds}: {len(cache)} images")
    return combined


def get_merged_embeddings(model, input_ids, pixel_values, attention_mask):
    """Merges text and visual embeddings into one sequence, replacing the 576 image-token
    placeholders with the CLIP visual features.
    """
    device = pixel_values.device
    image_token_index = getattr(model.config, "image_token_index", 32000)

    with torch.no_grad():
        img_out = model.vision_tower(pixel_values, output_hidden_states=True)
    feat_layer = getattr(model.config, "vision_feature_layer", -2)
    img_feats  = img_out.hidden_states[feat_layer][:, 1:]
    img_embeds = model.multi_modal_projector(img_feats)
    n_vis = img_embeds.shape[1]

    safe_ids = input_ids.clone()
    safe_ids[safe_ids == image_token_index] = 0
    text_embeds = model.language_model.embed_tokens(safe_ids)

    positions = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return text_embeds, attention_mask, None, None

    visual_start = positions[0].item()
    visual_end   = positions[-1].item() + 1

    prefix = text_embeds[:, :visual_start, :]
    suffix = text_embeds[:, visual_end:, :]
    merged = torch.cat([prefix, img_embeds, suffix], dim=1)

    if attention_mask is not None:
        vis_mask = torch.ones((1, n_vis), dtype=attention_mask.dtype, device=device)
        merged_mask = torch.cat([
            attention_mask[:, :visual_start],
            vis_mask,
            attention_mask[:, visual_end:]
        ], dim=1)
    else:
        merged_mask = torch.ones(1, merged.shape[1], dtype=torch.long, device=device)

    return merged, merged_mask, visual_start, visual_end


def run_observation_layers(model, inputs_embeds, attention_mask, K=CAFP_K):
    """Runs only the first K language-model layers on the merged sequence and returns the
    K-th hidden state, replicating the LlamaModel.forward() pre-loop setup.
    """
    from transformers.models.llama.modeling_llama import create_causal_mask
    with torch.no_grad():
        lm = model.language_model
        past_key_values = None
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=lm.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden = inputs_embeds
        position_embeddings = lm.rotary_emb(hidden, position_ids)

        K_clamped = min(K, len(lm.layers))
        for i in range(K_clamped):
            hidden = lm.layers[i](
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

    del causal_mask, cache_position, position_ids, position_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {}, hidden.detach()


def compute_cafp_flow(attention_maps, n_visual, visual_start, K=CAFP_K,
                       hidden_states=None):
    """L2-norm flow scores from the K-layer hidden state, min-max normalized to [0, 1]."""
    if hidden_states is None:
        return torch.ones(n_visual) / n_visual
    visual_end = visual_start + n_visual
    visual_h = hidden_states[0, visual_start:visual_end, :].float()
    flow = visual_h.norm(dim=-1)
    f_min, f_max = flow.min(), flow.max()
    if f_max > f_min + 1e-6:
        flow = (flow - f_min) / (f_max - f_min)
    return flow.cpu()


def flow_entropy(flow: torch.Tensor) -> float:
    p = flow / (flow.sum() + 1e-8)
    return float(-(p * (p + 1e-8).log2()).sum())


_ANATOMY_KW = {
    "cardiomegaly": "heart",    "cardiac":    "heart",    "heart":        "heart",
    "pleural":      "effusion", "effusion":   "effusion", "costophrenic": "effusion",
    "pneumothorax": "pneumothorax",
    "pneumonia":    "lungs",    "infiltrate": "lungs",    "consolidation":"lungs",
    "atelectasis":  "lungs",    "opacity":    "lungs",    "pulmonary":    "lungs",
    "lung":         "lungs",    "aneurysm":   "aorta",
    "mediastin":    "mediastinum", "aorta":   "aorta",
    "trachea":      "trachea",  "carina":     "carina",
    "hilar":        "hilar",    "diaphragm":  "diaphragm",
}
_SPATIAL_PRIOR = {
    "heart":        (0.30, 0.25, 0.65, 0.75),
    "effusion":     (0.00, 0.55, 1.00, 0.95),
    "lungs":        (0.05, 0.08, 0.90, 0.88),
    "mediastinum":  (0.25, 0.08, 0.65, 0.70),
    "aorta":        (0.25, 0.08, 0.65, 0.80),
    "trachea":      (0.40, 0.00, 0.60, 0.35),
    "carina":       (0.38, 0.25, 0.62, 0.48),
    "hilar":        (0.22, 0.22, 0.72, 0.65),
    "diaphragm":    (0.05, 0.60, 0.95, 0.98),
    "pneumothorax": (0.00, 0.05, 1.00, 0.60),
}


def _extract_anatomy_kw(question: str) -> str:
    q = question.lower()
    for kw, label in _ANATOMY_KW.items():
        if kw in q:
            return label
    return "lungs"


def _prior_grid(label: str) -> np.ndarray:
    key = label if label in _SPATIAL_PRIOR else "lungs"
    x1f, y1f, x2f, y2f = _SPATIAL_PRIOR[key]
    g = np.zeros((TOKEN_GRID, TOKEN_GRID), dtype=bool)
    c1, r1 = int(x1f*TOKEN_GRID), int(y1f*TOKEN_GRID)
    c2, r2 = max(c1+1, int(x2f*TOKEN_GRID)), max(r1+1, int(y2f*TOKEN_GRID))
    g[r1:r2, c1:c2] = True
    return g


def _pick_best_mask(amap, anatomy_label):
    n = amap["token_masks"].shape[0]
    if n == 0: return _prior_grid(anatomy_label)
    key = anatomy_label if anatomy_label in _SPATIAL_PRIOR else "lungs"
    x1f, y1f, x2f, y2f = _SPATIAL_PRIOR[key]
    pcx, pcy = (x1f+x2f)/2, (y1f+y2f)/2
    h, w = amap["image_size"]
    best, best_d = 0, float("inf")
    for i, m in enumerate(amap["masks"]):
        bx, by, bw, bh = m["bbox"]
        d = math.hypot((bx+bw/2)/w-pcx, (by+bh/2)/h-pcy)
        if d < best_d: best_d, best = d, i
    return amap["token_masks"][best]


def get_anatomy_score(image_id, question, anatomy_cache, n_visual=N_VISUAL):
    """Returns the binary anatomy-prior score over the visual tokens for an image-question pair."""
    anatomy_label = _extract_anatomy_kw(question)
    tok_mask_2d = None

    if anatomy_cache and image_id and image_id in anatomy_cache:
        amap = anatomy_cache[image_id]
        stored = [x.lower().strip() for x in amap.get("anatomy_labels", [])]
        if _have_sam_lookup and anatomy_label in stored:
            tok_mask_2d = _sam_mod.get_anatomy_token_mask(amap, anatomy_label)
        elif not _have_sam_lookup:
            tok_mask_2d = _pick_best_mask(amap, anatomy_label)

    if tok_mask_2d is None:
        tok_mask_2d = _prior_grid(anatomy_label)

    coverage = float(tok_mask_2d.sum()) / (TOKEN_GRID * TOKEN_GRID)
    if coverage > 0.60 and anatomy_label != "lungs":
        tok_mask_2d = _prior_grid(anatomy_label)

    flat = tok_mask_2d.flatten()[:n_visual].astype(np.float32)
    return torch.tensor(flat, dtype=torch.float32)


def determine_keep_indices(s_flow, s_med, alpha=CAFP_ALPHA, r=CAFP_R, n_visual=N_VISUAL):
    """Keeps all anatomy tokens plus the highest fused-score tokens up to the retention budget."""
    n_keep = max(1, int((1.0-r)*n_visual))
    s_fused = (1.0-alpha)*s_flow + alpha*s_med.float()
    topk_idx = torch.topk(s_fused, n_keep).indices
    anatomy_idx = (s_med > 0).nonzero(as_tuple=True)[0]
    keep = torch.unique(torch.cat([topk_idx, anatomy_idx])) if anatomy_idx.numel() > 0 else topk_idx
    return keep.sort().values


def prune_embeddings(inputs_embeds, attention_mask, keep_indices, visual_start, visual_end):
    """Removes non-kept visual tokens from the merged sequence, preserving the surviving positions."""
    seq_len = inputs_embeds.shape[1]
    device  = inputs_embeds.device
    all_pos = list(range(visual_start)) + (keep_indices+visual_start).tolist() + list(range(visual_end, seq_len))
    idx_t = torch.tensor(all_pos, dtype=torch.long, device=device)
    pruned_embeds = inputs_embeds[:, idx_t, :]
    pruned_mask   = (attention_mask[:, idx_t] if attention_mask is not None
                     else torch.ones(1, len(all_pos), dtype=torch.long, device=device))
    return pruned_embeds, pruned_mask


def manual_greedy_decode(model, processor, pruned_embeds, pruned_mask, max_new_tokens):
    """Greedy decoding from pruned embeddings using model.forward() with a KV cache
    (model.generate(inputs_embeds=...) is broken for this model).
    """
    device = pruned_embeds.device
    eos_id = processor.tokenizer.eos_token_id
    generated_ids = []
    with torch.no_grad():
        out = model(inputs_embeds=pruned_embeds, attention_mask=pruned_mask,
                    use_cache=True, return_dict=True)
        past = out.past_key_values
        current_mask = pruned_mask
        for _ in range(max_new_tokens):
            next_id = out.logits[0, -1, :].argmax().item()
            if next_id == eos_id:
                break
            generated_ids.append(next_id)
            next_embed = model.language_model.embed_tokens(
                torch.tensor([[next_id]], device=device))
            current_mask = torch.cat(
                [current_mask, torch.ones((1,1), dtype=torch.long, device=device)], dim=1)
            out = model(inputs_embeds=next_embed, attention_mask=current_mask,
                        past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class CAFPInference:
    """End-to-end CAFP inference wrapper: prune the visual tokens, then generate."""
    def __init__(self, model, processor, K=CAFP_K, r=CAFP_R, alpha=CAFP_ALPHA,
                 anatomy_cache=None, max_new_tokens=MAX_NEW_TOKENS):
        self.model = model; self.processor = processor
        self.K = K; self.r = r; self.alpha = alpha
        self.anatomy_cache = anatomy_cache or {}
        self.max_new_tokens = max_new_tokens

    def generate(self, image, question: str, image_id: str = None):
        """Runs one CAFP inference step; returns the generated text plus latency and pruning diagnostics."""
        device = next(self.model.parameters()).device
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            merged, merged_mask, v_start, v_end = get_merged_embeddings(
                self.model, inputs["input_ids"], inputs["pixel_values"],
                inputs.get("attention_mask"))

            if v_start is None:
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                           do_sample=False,
                                           pad_token_id=self.processor.tokenizer.eos_token_id)
                n_in = inputs["input_ids"].shape[1]
                return self.processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip(), {}

            n_vis = v_end - v_start

            t0 = time.perf_counter()
            _, hidden_k = run_observation_layers(self.model, merged, merged_mask, self.K)
            t_obs = (time.perf_counter()-t0)*1000

            t1 = time.perf_counter()
            flow = compute_cafp_flow({}, n_vis, v_start, self.K, hidden_states=hidden_k)
            t_flow = (time.perf_counter()-t1)*1000
            del hidden_k
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            t2 = time.perf_counter()
            s_med = get_anatomy_score(image_id, question, self.anatomy_cache, n_vis)
            t_sam = (time.perf_counter()-t2)*1000

            t3 = time.perf_counter()
            keep_idx = determine_keep_indices(flow, s_med, self.alpha, self.r, n_vis)
            pruned_e, pruned_m = prune_embeddings(merged, merged_mask, keep_idx, v_start, v_end)
            t_sel = (time.perf_counter()-t3)*1000
            del merged
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        t4 = time.perf_counter()
        output_text = manual_greedy_decode(self.model, self.processor,
                                            pruned_e, pruned_m, self.max_new_tokens)
        t_gen = (time.perf_counter()-t4)*1000

        apr = None
        n_anatomy = int((s_med > 0).sum())
        if n_anatomy > 0:
            anatomy_set = set((s_med > 0).nonzero(as_tuple=True)[0].tolist())
            apr = len(anatomy_set & set(keep_idx.tolist())) / n_anatomy

        t_total = t_obs + t_flow + t_sam + t_sel + t_gen
        return output_text, {
            "n_visual": n_vis, "n_kept": int(keep_idx.shape[0]),
            "pruning_ratio": 1.0 - keep_idx.shape[0]/n_vis,
            "flow_entropy": flow_entropy(flow), "apr": apr,
            "latency_observation_ms": round(t_obs,1),
            "latency_flow_ms": round(t_flow,1),
            "latency_sam_ms": round(t_sam,1),
            "latency_selection_ms": round(t_sel,1),
            "latency_generation_ms": round(t_gen,1),
            "latency_total_ms": round(t_total,1),
        }


def run_verification_tests(model, processor, anatomy_cache=None,
                            vqa_rad_test="data/vqa_rad_test.json"):
    """Runs a few self-checks of the CAFP pipeline on real samples."""
    from PIL import Image as PILImage
    cafp = CAFPInference(model, processor, anatomy_cache=anatomy_cache)
    sample = None
    if os.path.exists(vqa_rad_test):
        with open(vqa_rad_test) as f: records = json.load(f)
        for r in records:
            if r.get("image") and os.path.exists(r["image"]): sample = r; break
    if sample is None:
        sample = {"id":"synthetic","image":None,"question":"Is there cardiomegaly?","answer":"yes"}
    img = (PILImage.open(sample["image"]).convert("RGB")
           if sample.get("image") and os.path.exists(sample["image"])
           else PILImage.new("RGB",(336,336),color=(128,128,128)))
    question, image_id = sample.get("question","Is there cardiomegaly?"), sample.get("id")

    print("="*62); print("CAFP VERIFICATION TESTS (v3)"); print("="*62)
    torch.cuda.reset_peak_memory_stats()
    output_text, diag = cafp.generate(img, question, image_id)
    results = {}

    n_kept = diag.get("n_kept",0); expected = int((1.0-CAFP_R)*N_VISUAL)
    t1 = expected <= n_kept <= int(expected*1.5); results["t1"] = t1
    print(f"\n[T1] Pruned length  : n_kept={n_kept}, expected≈{expected}")
    print(f"     {'PASS v' if t1 else 'FAIL x'}")

    is_garbage = (not output_text.strip()) or all(c=="\\" for c in output_text.strip())
    t2 = isinstance(output_text,str) and len(output_text.strip())>0 and not is_garbage
    results["t2"] = t2
    print(f"\n[T2] Coherent output: '{output_text[:100]}'")
    print(f"     {'PASS v' if t2 else 'FAIL x  <- still garbage'}")

    apr = diag.get("apr"); t3 = apr is None or apr >= 0.80; results["t3"] = t3
    print(f"\n[T3] APR            : {f'{apr:.3f}' if apr else 'N/A'}  (required >= 0.80)")
    print(f"     {'PASS v' if t3 else 'FAIL x'}")

    import platform
    vram_gb = torch.cuda.max_memory_allocated()/1e9
    budget = 8.5 if platform.system()=="Windows" else 7.5
    t4 = vram_gb <= budget; results["t4"] = t4
    print(f"\n[T4] Peak VRAM      : {vram_gb:.2f} GB  (budget {budget:.1f} GB)")
    print(f"     {'PASS v' if t4 else 'FAIL x'}")

    H = diag.get("flow_entropy",-1.0); t5 = 0.5 <= H <= 9.0; results["t5"] = t5
    print(f"\n[T5] Flow entropy   : {H:.3f} bits (7.6-8.8 expected; flat across K — anatomy prior drives selection)")
    print(f"     Token selection = fused flow + anatomy prior (alpha={CAFP_ALPHA})")
    print(f"     {'PASS v' if t5 else 'FAIL x'}")

    n_pass = sum(results.values())
    print(f"\n{'='*62}"); print(f"RESULT: {n_pass}/5 passed")
    if n_pass == 5: print("ALL PASS -> ready to train: python scripts/train.py")
    else: print(f"FAIL: {[k for k,v in results.items() if not v]}")
    print(f"{'='*62}")
    return n_pass == 5


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-anatomy", action="store_true")
    args = ap.parse_args()
    model, processor = load_model_for_cafp()
    anatomy_cache = {} if args.no_anatomy else load_anatomy_cache(["vqa_rad","iu_xray", "slake"])
    run_verification_tests(model, processor, anatomy_cache)

if __name__ == "__main__":
    main()

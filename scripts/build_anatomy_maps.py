import os, sys, json, pickle, time, argparse
import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SAM_DIR    = "SAM-Med2D"
CHECKPOINT = "checkpoints/sam_med2d/sam-med2d_b.pth"
CACHE_DIR  = "data/anatomy_maps_cache"
DATA_DIR   = "data"

LLAVA_SIZE = 336
TOKEN_GRID = 24
PATCH_PX   = LLAVA_SIZE // TOKEN_GRID


MASK_SOURCE_GT = "slake_gt"
MASK_SOURCE_SAM = "sam"
MASK_SOURCE_PRIOR = "prior"
MASK_SOURCE_UNKNOWN = "unknown"


CHEST_XRAY_TARGETS = {
    "heart": (0.50,0.60),
    "left_lung": (0.30,0.45),
    "right_lung": (0.70,0.45),
    "pleura": (0.50,0.85),
    "mediastinum": (0.50,0.30),
}

ANATOMY_KW = {

    "heart": "heart",
    "cardiac": "heart",
    "cardiomegaly": "heart",
    "cardiomediastinal": "heart",
    "ventricle": "heart",
    "ventricular": "heart",
    "atrium": "heart",
    "atrial": "heart",
    "myocardial": "heart",

    "lung": "lung",
    "pulmonary": "lung",
    "pneumonia": "lung",
    "consolidation": "lung",
    "opacity": "lung",
    "opacities": "lung",
    "infiltrate": "lung",
    "atelectasis": "lung",
    "cxr": "lung",
    "chest": "lung",
    "thoracic": "lung",

    "pleural": "pleura",
    "effusion": "pleura",
    "pneumothorax": "pleura",
    "costophrenic": "pleura",

    "mediastinum": "mediastinum",
    "mediastinal": "mediastinum",
    "hilum": "mediastinum",
    "hilar": "mediastinum",

    "liver": "liver",
    "hepatic": "liver",

    "kidney": "kidney",
    "renal": "kidney",

    "brain": "brain",
    "cerebral": "brain",
    "intracranial": "brain",

    "spine": "spine",
    "spinal": "spine",
    "vertebra": "spine",
    "vertebral": "spine",

    "abdomen": "abdomen",
    "abdominal": "abdomen",

    "bone": "bone",
    "osseous": "bone",
    "skeletal": "bone",
}

CHEST_ORGANS = {
    "heart",
    "lung",
    "pleura",
    "mediastinum",
}
SLAKE_TARGETS = {
    "brain": (0.50, 0.35),
    "spine": (0.50, 0.55),
    "abdomen": (0.50, 0.72),
    "liver": (0.60, 0.70),
    "kidney": (0.45, 0.72),
    "bone": (0.50, 0.50),
}


DATASETS = {
    "vqa_rad": {
        "files":  ["vqa_rad_train.json", "vqa_rad_test.json"],
        "output": "vqa_rad_maps.pkl",
    },
    "iu_xray": {
        "files":  ["iu_xray_train.json", "iu_xray_test.json"],
        "output": "iu_xray_maps.pkl",
    },
    "slake": {
        "files": ["slake_train_en.json", "slake_test_en.json"],
        "output": "slake_maps.pkl",
    }
}

def get_dataset_targets(dataset_name):
    if dataset_name in ["vqa_rad", "iu_xray"]:
        return CHEST_XRAY_TARGETS

    if dataset_name == "slake":
        return SLAKE_TARGETS

    return CHEST_XRAY_TARGETS


def load_sam_predictor(checkpoint: str, device: str):
    """Loads SAM-Med2D from the local clone and returns a SamPredictor."""
    if os.path.isdir(SAM_DIR):
        sys.path.insert(0, os.path.abspath(SAM_DIR))
        print(f"[SAM] Using SAM-Med2D from ./{SAM_DIR}/")
    else:
        sys.exit(
            f"ERROR: {SAM_DIR}/ not found.\n"
            "Clone: git clone https://github.com/uni-medical/SAM-Med2D"
        )

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        sys.exit(
            "ERROR: segment_anything not importable.\n"
            "Run: pip install -r SAM-Med2D/requirements.txt --break-system-packages"
        )

    from types import SimpleNamespace
    try:
        sam_args = SimpleNamespace(
            image_size=256,
            sam_checkpoint=checkpoint,
            encoder_adapter=True,
        )

        sam = sam_model_registry["vit_b"](sam_args)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint}: {e}"
        )


    if not os.path.exists(checkpoint):
        sys.exit(
            f"ERROR: checkpoint not found: {checkpoint}\n"
            'Run: python -c "from huggingface_hub import hf_hub_download; '
            "hf_hub_download('wangrongsheng/SAM-Med2D', 'sam-med2d_b.pth', "
            "local_dir='checkpoints/sam_med2d')\" "
        )

    sam.eval().to(device)
    predictor = SamPredictor(sam)
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"[SAM] SamPredictor loaded. VRAM: {vram:.2f} GB")
    return predictor


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load any image (grayscale or RGB) → uint8 RGB (H, W, 3)."""
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    from PIL import Image as PILImage
    return np.array(PILImage.open(image_path).convert("RGB"), dtype=np.uint8)

def mask_to_token_grid(mask_hw: np.ndarray, src_h: int, src_w: int) -> np.ndarray:
    """
    Remap a binary (H, W) mask → (24, 24) bool token grid.
    Token (r, c) = True if > 50% of its 14×14 pixel patch overlaps the mask.
    """
    try:
        import cv2
        resized = cv2.resize(
            mask_hw.astype(np.uint8) * 255,
            (LLAVA_SIZE, LLAVA_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
    except Exception:
        from PIL import Image as PILImage
        resized = np.array(
            PILImage.fromarray(mask_hw.astype(np.uint8) * 255).resize(
                (LLAVA_SIZE, LLAVA_SIZE), resample=0
            ),
            dtype=np.uint8,
        )
    grid = np.zeros((TOKEN_GRID, TOKEN_GRID), dtype=bool)
    for r in range(TOKEN_GRID):
        for c in range(TOKEN_GRID):
            patch = resized[r*PATCH_PX:(r+1)*PATCH_PX, c*PATCH_PX:(c+1)*PATCH_PX]
            grid[r, c] = patch.mean() > 127.5
    return grid

def is_valid_mask(
    mask: np.ndarray,
    score: float,
    min_score: float = 0.60,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.85,
):
    """Rejects degenerate SAM masks: tiny specks, near-full-image masks, or low-confidence predictions."""

    area_ratio = mask.mean()

    if score < min_score:
        return False

    if area_ratio < min_area_ratio:
        return False

    if area_ratio > max_area_ratio:
        return False

    return True


def process_one_image(
    image_path: str,
    image_id: str,
    predictor,
    anatomy_targets: dict = CHEST_XRAY_TARGETS,
) -> dict:
    """
    Run SamPredictor once per anatomy region using a center-point prompt.
    SAM returns 3 mask candidates; we keep the one with highest predicted IoU.
    """
    img_rgb = load_image_rgb(image_path)
    orig_h, orig_w = img_rgb.shape[:2]

    predictor.set_image(img_rgb)

    masks_meta  = []
    token_masks = []
    labels_out  = []

    for anatomy_label, (cx_frac, cy_frac) in anatomy_targets.items():
        px = int(cx_frac * orig_w)
        py = int(cy_frac * orig_h)

        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[px, py]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )

        best = int(np.argmax(scores))
        best_mask  = masks[best]
        best_score = float(scores[best])
        if not is_valid_mask(best_mask, best_score):
            continue

        rows = np.any(best_mask, axis=1)
        cols = np.any(best_mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox = [int(cmin), int(rmin), int(cmax-cmin), int(rmax-rmin)]
        else:
            bbox = [0, 0, 0, 0]

        tok_grid = mask_to_token_grid(best_mask, orig_h, orig_w)

        masks_meta.append({
            "area":          int(best_mask.sum()),
            "area_ratio":    float(best_mask.mean()),
            "bbox":          bbox,
            "iou_pred":      best_score,
            "anatomy_label": anatomy_label,
            "mask_source": MASK_SOURCE_SAM,
        })
        token_masks.append(tok_grid)
        labels_out.append(anatomy_label)

    return {
        "image_id":       image_id,
        "image_size":     (orig_h, orig_w),
        "masks":          masks_meta,
        "mask_source":    MASK_SOURCE_SAM,
        "token_masks":    np.array(token_masks, dtype=bool),
        "anatomy_labels": labels_out,
    }


def process_one_slake_image(image_path, image_id):

    image_path = Path(image_path)

    detection_file = image_path.parent / "detection.json"

    if not detection_file.exists():
        return {
            "image_id": image_id,
            "image_size": (0, 0),
            "masks": [],
            "mask_source": MASK_SOURCE_UNKNOWN,
            "token_masks": np.zeros((0, 24, 24), dtype=bool),
            "anatomy_labels": [],
        }

    img = load_image_rgb(str(image_path))
    H, W = img.shape[:2]

    with open(detection_file, "r", encoding="utf-8") as f:
        detections = json.load(f)

    token_masks = []
    masks_meta = []
    labels_out = []

    for item in detections:

        for organ_name, bbox in item.items():

            x, y, bw, bh = bbox

            x = int(x)
            y = int(y)
            bw = int(bw)
            bh = int(bh)

            binary = np.zeros((H, W), dtype=bool)

            x2 = min(x + bw, W)
            y2 = min(y + bh, H)

            binary[y:y2, x:x2] = True

            tok = mask_to_token_grid(binary, H, W)

            token_masks.append(tok)

            masks_meta.append({
                "area": int(binary.sum()),
                "area_ratio": float(binary.mean()),
                "bbox": [x, y, bw, bh],
                "iou_pred": 1.0,
                "anatomy_label": organ_name.lower(),
                "mask_source": MASK_SOURCE_GT,
            })

            labels_out.append(organ_name.lower())

    return {
        "image_id": image_id,
        "image_size": (H, W),
        "masks": masks_meta,
        "mask_source": MASK_SOURCE_GT,
        "token_masks": np.array(token_masks, dtype=bool),
        "anatomy_labels": labels_out,
    }


def extract_anatomy_label(question, available_labels=None):

    q = question.lower()

    if available_labels:
        for organ in available_labels:
            if organ.lower() in q:
                return organ.lower()

    for kw, label in ANATOMY_KW.items():
        if kw in q:
            return label

    return "unknown"


def get_anatomy_token_mask(anatomy_map: dict, anatomy_term: str) -> np.ndarray:
    """
    Return the (24, 24) bool token mask for anatomy_term from a stored map.
    Falls back to spatial prior if the label is not found.
    """
    label = anatomy_term.lower().strip()
    for kw, mapped in ANATOMY_KW.items():
        if kw in label:
            label = mapped
            break

    stored = [
        x.lower().strip()
        for x in anatomy_map.get("anatomy_labels", [])
    ]
    if label in stored:
        idx = stored.index(label)
        return anatomy_map["token_masks"][idx]

    if label == "unknown":
        return np.ones((TOKEN_GRID, TOKEN_GRID), dtype=bool)

    return _prior_to_grid(label)


def _prior_to_grid(label: str) -> np.ndarray:
    PRIORS = {
        "heart":        (0.28, 0.28, 0.62, 0.72),
        "lung":        (0.03, 0.06, 0.97, 0.88),
        "pleura":     (0.00, 0.62, 1.00, 0.97),
        "mediastinum": (0.30, 0.10, 0.70, 0.55),
    }
    key = label if label in PRIORS else "lung"
    x1f, y1f, x2f, y2f = PRIORS[key]
    g = np.zeros((TOKEN_GRID, TOKEN_GRID), dtype=bool)
    c1, r1 = int(x1f * TOKEN_GRID), int(y1f * TOKEN_GRID)
    c2, r2 = max(c1+1, int(x2f*TOKEN_GRID)), max(r1+1, int(y2f*TOKEN_GRID))
    g[r1:r2, c1:c2] = True
    return g


def resolve_image_path(img_field: str) -> str | None:
    for p in [img_field, os.path.join(DATA_DIR, img_field),
              os.path.join(DATA_DIR, "images", img_field)]:
        if p and os.path.isfile(p):
            return p
    return None


def batch_process(
    dataset_name: str,
    json_files: list,
    output_pkl: str,
    predictor,
    dry_run: bool = False,
) -> dict:
    out_path = os.path.join(CACHE_DIR, output_pkl)
    cache = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[{dataset_name}] Resuming: {len(cache)} already cached")

    all_pairs = {}
    for fname in json_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping {fname} (not found)")
            continue
        with open(fpath) as f:
            records = json.load(f)
        for i, r in enumerate(records):
            iid   = r.get("id") or r.get("image_id") or f"{dataset_name}_{i}"
            ipath = r.get("image", "")
            if iid not in all_pairs:
                all_pairs[iid] = ipath

    todo = [(iid, p) for iid, p in all_pairs.items() if iid not in cache]
    if dry_run:
        todo = todo[:5]
        print(f"[{dataset_name}] DRY RUN: {len(todo)} images")
    else:
        print(f"[{dataset_name}] {len(all_pairs)} unique images, "
              f"{len(cache)} cached, {len(todo)} to process")

    errors, save_every = 0, 200
    for idx, (iid, img_field) in enumerate(tqdm(todo, desc=dataset_name)):
        full_path = resolve_image_path(img_field)
        if full_path is None:
            errors += 1
            continue
        try:
            targets = get_dataset_targets(dataset_name)
            if dataset_name == "slake":
                result = process_one_slake_image(full_path, iid,)
            else:
                result = process_one_image(full_path, iid, predictor, anatomy_targets=targets,)

            cache[iid] = result
        except Exception as e:
            print(f"\n  ERROR {iid}: {e}")
            errors += 1
            continue
        if (idx + 1) % save_every == 0:
            with open(out_path, "wb") as f:
                pickle.dump(cache, f, protocol=4)

    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{dataset_name}] Done. {len(cache)} images, "
          f"{errors} errors. Saved {size_mb:.0f} MB → {out_path}")
    return cache


def print_cache_summary(cache: dict, name: str):
    n     = len(cache)
    n_ok  = sum(1 for v in cache.values() if v["token_masks"].shape[0] > 0)
    print(f"\n[{name}] Summary:")
    print(f"  Images:             {n}")
    print(f"  With masks:         {n_ok}")
    print(f"  Anatomy regions/img: {len(CHEST_XRAY_TARGETS)}")

    if len(cache)==0:
        print("Empty cache")
        return 

    sample = next(iter(cache.values()))
    for i, label in enumerate(sample.get("anatomy_labels", [])):
        covs = []
        for v in list(cache.values())[:100]:
            if i < v["token_masks"].shape[0]:
                covs.append(v["token_masks"][i].mean() * 100)
        if covs:
            print(f"  {label}: avg coverage = {np.mean(covs):.1f}% "
                  f"({np.mean(covs)/100*576:.0f} / 576 tokens)")

    print(f"\n  Routing test (first image):")
    for q in ["Is there cardiomegaly?", "Is there a pleural effusion?",
              "Is there pneumothorax?"]:
        label = extract_anatomy_label(q, sample["anatomy_labels"])

        tok   = get_anatomy_token_mask(sample, label)
        print(f"    '{q}' → '{label}' → {tok.sum()} tokens")

def compute_mask_statistics(cache):
    """
    Useful for paper reporting.
    """

    total_masks = 0
    sam_masks = 0
    prior_masks = 0
    avg_area = []


    for item in cache.values():

        total_masks += len(item["masks"])

        for m in item["masks"]:
            if m["mask_source"] == MASK_SOURCE_SAM:
                sam_masks += 1
            if "area_ratio" in m:
                avg_area.append(m["area_ratio"])
            else:
                prior_masks += 1
    if avg_area:
        print(f"Average mask area: {100*np.mean(avg_area):.2f}%")

    print("\nMask Statistics")
    print(f"Total masks : {total_masks}")
    print(f"SAM masks   : {sam_masks}")
    print(f"Prior masks : {prior_masks}")
    print(f"SAM usage   : {100*sam_masks/max(total_masks,1):.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", choices=["vqa_rad", "iu_xray", "slake"])
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({p.total_memory/1e9:.1f} GB)")

    predictor = load_sam_predictor(CHECKPOINT, device)

    t0 = time.time()
    datasets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    for name, cfg in datasets.items():
        cache = batch_process(name, cfg["files"], cfg["output"],
                               predictor, dry_run=args.dry_run)

        compute_mask_statistics(cache)
        print_cache_summary(cache, name)

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal time: {elapsed:.1f} min")

    if args.dry_run:
        print("\nIf routing test shows non-zero token counts → full run:")
        print("  uv run scripts/build_anatomy_maps.py")
        print("Expected: ~20-30 min (10x faster than automatic generation)")
    else:
        print("\nAnatomy-map cache complete.")


if __name__ == "__main__":
    main()

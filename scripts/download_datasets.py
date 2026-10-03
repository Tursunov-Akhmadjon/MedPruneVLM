"""Downloads VQA-RAD from HuggingFace, saves images to data/vqa_rad_images/ and
writes the unified train/test JSONs. SLAKE and IU X-Ray images are downloaded manually
"""
import os, json, random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

random.seed(42)


def download_vqa_rad():
    print("\n=== VQA-RAD ===")
    from datasets import load_dataset

    try:
        ds = load_dataset("flaviagiammarino/vqa-rad")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fix connection, then re-run.")
        return False

    print(f"  Splits   : {list(ds.keys())}")
    print(f"  Columns  : {ds['train'].column_names}")
    first_row = {k: v for k, v in dict(list(ds["train"])[0]).items() if k != "image"}
    print(f"  Row 0    : {json.dumps(first_row, indent=4)}")

    results = {"train": [], "test": []}

    for split in ["train", "test"]:
        img_dir = f"data/vqa_rad_images/{split}"
        os.makedirs(img_dir, exist_ok=True)
        n = len(ds[split])
        print(f"\n  Processing {split} ({n} samples)…")

        for i, row in enumerate(tqdm(ds[split], desc=f"  VQA-RAD/{split}")):
            img_path = f"{img_dir}/{i:04d}.jpg"
            if not os.path.exists(img_path):
                try:
                    row["image"].convert("RGB").save(img_path)
                except Exception as ex:
                    img_path = None
                    print(f"    warn: img {i} not saved: {ex}")

            ans = str(row.get("answer", "")).strip()
            is_yn = ans.lower() in ("yes", "no")

            entry = {
                "id": f"vqarad_{split}_{i:04d}",
                "image": img_path,
                "question": str(row.get("question", "")).strip(),
                "answer": ans,
                "dataset": "vqa_rad",
                "split": split,
                "cross_image_neg": False,
                "anatomy_region": None,
                "question_type": "yes_no" if is_yn else "other",
            }
            for k, v in row.items():
                if k not in ("image",) and k not in entry:
                    entry[k] = v

            results[split].append(entry)

    with open("data/vqa_rad_train.json", "w") as f:
        json.dump(results["train"], f)
    with open("data/vqa_rad_test.json", "w") as f:
        json.dump(results["test"], f)

    n_yn = sum(1 for s in results["test"] if s["question_type"] == "yes_no")
    print(f"\n  ✓ train={len(results['train'])} | test={len(results['test'])} | yes/no_test={n_yn}")

    if len(results["test"]) < 400:
        print(f"  ⚠ Expected ~451 test samples, got {len(results['test'])}")
    if n_yn < 200:
        print(f"  ⚠ Expected ~263 yes/no in test, got {n_yn}")

    return True


if __name__ == "__main__":
    print("=== Dataset Download ===")
    vqa_ok = download_vqa_rad()
    print(
        "\nSLAKE and IU X-Ray images are NOT downloaded automatically:\n"
        "  SLAKE    : download from https://github.com/Med-AIUJ/SLAKE and extract\n"
        "             the archive so images live at  data/images/slake/imgs/\n"
        "  IU X-Ray : download the OpenI Indiana University chest X-ray set and\n"
        "             extract so images live at  data/images/iu_xray/data/images/\n"
        "             and indiana_dataset.csv at  data/images/iu_xray/data/\n"
    )
    print(f"\n=== Done === VQA-RAD: {vqa_ok}")
    print("Next: python scripts/build_pope_med.py")

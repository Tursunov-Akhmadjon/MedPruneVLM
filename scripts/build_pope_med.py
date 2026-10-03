"""Builds the static POPE-Med benchmark from VQA-RAD and SLAKE yes/no questions with
cross-image hard negatives, and writes the extended training set (seed 42).
"""
import json, random, os
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

print("=== Building POPE-Med Static Benchmark ===\n")

test_data = []

with open("data/vqa_rad_test.json") as f:
    test_data.extend(json.load(f))

with open("data/slake_test_en.json") as f:
    slake_test = json.load(f)

test_data.extend([s for s in slake_test if s["question_type"] == "yes_no"])

yn_samples = [s for s in test_data if s["question_type"] == "yes_no"]
n_yes = sum(1 for s in yn_samples if s["answer"].lower() == "yes")
n_no  = len(yn_samples) - n_yes
print(f"Test yes/no pool : {len(yn_samples)}  (Yes={n_yes}, No={n_no})")
print(f"Yes rate         : {100*n_yes/max(len(yn_samples),1):.1f}%")


img_yes_findings = defaultdict(list)
img_no_findings  = defaultdict(list)

for s in yn_samples:
    img_id = s["image"]
    if s["answer"].lower() == "yes":
        img_yes_findings[img_id].append(s)
    else:
        img_no_findings[img_id].append(s)

all_images = sorted(img_yes_findings.keys() | img_no_findings.keys())
print(f"Unique test images: {len(all_images)}")


print("\nGenerating cross-image hard negatives…")
cross_neg_samples = []
cross_neg_audit   = []

for target_img in tqdm(all_images, desc="  cross-image neg"):
    other_candidates = []
    for other_img, findings in img_yes_findings.items():
        if other_img == target_img:
            continue
        for s in findings:
            other_candidates.append({"source_img": other_img, "sample": s})

    if not other_candidates:
        continue

    random.shuffle(other_candidates)
    selected, used_sources = [], set()
    for cand in other_candidates:
        if cand["source_img"] not in used_sources and len(selected) < 1:
            selected.append(cand)
            used_sources.add(cand["source_img"])
        if len(selected) == 1:
            break

    for cand in selected:
        neg = {
            "id": f"pope_xn_{len(cross_neg_samples):05d}",
            "image": target_img,
            "question": cand["sample"]["question"],
            "answer": "No",
            "dataset": "pope_med",
            "split": "pope_med_eval",
            "task_type": cand["sample"].get("task_type"),
            "question_type": "yes_no",
            "cross_image_neg": True,
            "anatomy_region": cand["sample"].get("anatomy_region"),
            "metadata": cand["sample"].get("metadata", {}),
        }
        cross_neg_samples.append(neg)
        cross_neg_audit.append({
            "sample_id": neg["id"],
            "target_image": target_img,
            "source_image": cand["source_img"],
            "question": cand["sample"]["question"],
        })

print(f"  Cross-image negatives: {len(cross_neg_samples)}")


original_negs = [dict(s) for s in yn_samples if s["answer"].lower() == "no"]
positives     = [dict(s) for s in yn_samples if s["answer"].lower() == "yes"]
positives_sample = positives

for s in original_negs + positives_sample:
    s["split"] = "pope_med_eval"
    s["cross_image_neg"] = False

pope_med = original_negs + cross_neg_samples + positives_sample
random.shuffle(pope_med)

total   = len(pope_med)
tot_neg = sum(1 for s in pope_med if s["answer"].lower() == "no")
tot_pos = total - tot_neg
tot_xn  = sum(1 for s in pope_med if s["cross_image_neg"])
print(f"\nPOPE-Med final stats:")
print(f"  Total          : {total}")
print(f"  Negative (No)  : {tot_neg} ({100*tot_neg/total:.1f}%)")
print(f"  Positive (Yes) : {tot_pos} ({100*tot_pos/total:.1f}%)")
print(f"  Cross-img neg  : {tot_xn}")

with open("data/pope_med_eval.json", "w") as f:
    json.dump(pope_med, f, indent=2, ensure_ascii=False)
with open("data/cross_image_neg_pairs.json", "w") as f:
    json.dump(cross_neg_audit, f, indent=2, ensure_ascii=False)
print("\n  ✓ data/pope_med_eval.json  (STATIC — do not modify)")
print("  ✓ data/cross_image_neg_pairs.json")


print("\n=== Building Training Data with Cross-Image Negatives ===")

train_data = []

with open("data/vqa_rad_train.json") as f:
    train_data.extend(json.load(f))

with open("data/slake_train_en.json") as f:
    train_data.extend([s for s in json.load(f) if s["question_type"] == "yes_no"])

train_yn   = [s for s in train_data if s["question_type"] == "yes_no"]
train_imgs = sorted(set(s["image"] for s in train_yn))

train_img_yes = defaultdict(list)
for s in train_yn:
    if s["answer"].lower() == "yes":
        train_img_yes[s["image"]].append(s)

print(f"Training yes/no pool : {len(train_yn)}  unique images : {len(train_imgs)}")
print("Generating training cross-image negatives (2 per image)…")

train_cross_neg = []
for target_img in tqdm(train_imgs, desc="  train cross-neg"):
    candidates = []
    for other_img, findings in train_img_yes.items():
        if other_img == target_img:
            continue
        for s in findings:
            candidates.append({"source_img": other_img, "sample": s})

    if not candidates:
        continue

    random.shuffle(candidates)
    selected, used = [], set()
    for cand in candidates:
        if cand["source_img"] not in used and len(selected) < 1:
            selected.append(cand)
            used.add(cand["source_img"])
        if len(selected) == 1:
            break

    for cand in selected:
        train_cross_neg.append({
            "id": f"train_xn_{len(train_cross_neg):05d}",
            "image": target_img,
            "question": cand["sample"]["question"],
            "answer": "No",
            "dataset": "cross_neg",
            "split": "train",
            "task_type": cand["sample"].get("task_type"),
            "question_type": "yes_no",
            "cross_image_neg": True,
            "anatomy_region": cand["sample"].get("anatomy_region"),
            "metadata": cand["sample"].get("metadata", {})
        })

for s in train_data:
    s.setdefault("cross_image_neg", False)

slake_train = []
if os.path.exists("data/slake_train_en.json"):
    with open("data/slake_train_en.json") as f:
        slake_train = json.load(f)

iu_xray_train = []
if os.path.exists("data/iu_xray_train.json"):
    with open("data/iu_xray_train.json") as f:
        iu_xray_train = json.load(f)

combined_train = train_data + slake_train + iu_xray_train + train_cross_neg
with open("data/training_data_with_cross_neg.json", "w") as f:
    json.dump(combined_train, f, indent=2, ensure_ascii=False)

total_train   = len(combined_train)
tot_neg = sum(1 for s in combined_train if s["answer"].lower() == "no")
tot_pos = total_train - tot_neg
print(f"\nTraining data final stats:")
print(f"  Total          : {total_train}")
print(f"  Negative (No)  : {tot_neg} ({100*tot_neg/total_train:.1f}%)")
print(f"  Positive (Yes) : {tot_pos} ({100*tot_pos/total_train:.1f}%)")

print(f"\n  VQA-RAD orig  : {len(train_data)}")
print(f"  SLAKE EN orig : {len(slake_train)}")
print(f"  IU X-Ray orig : {len(iu_xray_train)}")
print(f"  Cross-neg     : {len(train_cross_neg)}")
print(f"  TOTAL         : {len(combined_train)}")
print("\n  ✓ data/training_data_with_cross_neg.json")
print("\nBenchmark and training set rebuilt.")

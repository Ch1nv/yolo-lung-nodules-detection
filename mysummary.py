from pathlib import Path
from collections import Counter

base_dir = Path(r"D:\deeplearning\lung_nodules_od\ct_images\labels")

class_names = {
    0: "Nodules_type1",
    1: "Nodules_type2"
}

def count_labels(label_dir):
    counter = Counter()

    for txt_file in label_dir.glob("*.txt"):
        if txt_file.name == "classes.txt":
            continue

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_id = int(line.split()[0])
                counter[class_id] += 1

    return counter

for split in ["train", "val"]:
    label_dir = base_dir / split
    counts = count_labels(label_dir)

    print(f"\n=== {split} ===")
    for class_id, class_name in class_names.items():
        print(f"{class_name}: {counts[class_id]}")
        
# === train ===
# Nodules_type1: 168
# Nodules_type2: 93

# === val ===
# Nodules_type1: 36
# Nodules_type2: 8
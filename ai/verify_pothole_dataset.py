from pathlib import Path

DATASET = Path(r"D:\smart-road-damage-system\ai\pothole_dataset")

splits = {
    "train": 700,
    "val": 200,
    "test": 100
}

print("=" * 60)
print("PAVE SPOT - FINAL DATASET VERIFICATION")
print("=" * 60)

total_images = 0
total_labels = 0

for split, expected in splits.items():

    image_folder = DATASET / split / "images"
    label_folder = DATASET / split / "labels"

    images = list(image_folder.glob("*"))
    labels = list(label_folder.glob("*.txt"))

    image_stems = {x.stem for x in images}
    label_stems = {x.stem for x in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    pothole_only = True
    invalid_labels = []

    for label_file in labels:
        lines = label_file.read_text(encoding="utf-8").splitlines()

        for line in lines:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5 or parts[0] != "0":
                pothole_only = False
                invalid_labels.append(
                    f"{label_file.name}: {line}"
                )

    print(f"\n{split.upper()}")
    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")
    print(f"Expected     : {expected}")

    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Pothole only : {pothole_only}")

    total_images += len(images)
    total_labels += len(labels)

print("\n" + "=" * 60)
print("TOTAL")
print("=" * 60)

print(f"Images: {total_images}")
print(f"Labels: {total_labels}")

yaml_file = DATASET / "data.yaml"

print(f"\ndata.yaml exists: {yaml_file.exists()}")

if (
    total_images == 1000
    and total_labels == 1000
    and pothole_only
    and yaml_file.exists()
):
    print("\nSUCCESS!")
    print("Dataset is ready for YOLO training.")
else:
    print("\nWARNING!")
    print("Please check the results above.")
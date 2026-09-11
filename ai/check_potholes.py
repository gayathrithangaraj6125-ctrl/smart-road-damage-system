from pathlib import Path

# Original RDD2022 dataset location
DATASET_PATH = Path(r"C:\Users\stgrj\Downloads\archive (7)")

# Pothole class ID from data.yaml
POTHOLE_CLASS_ID = "3"


def find_pothole_images(split):
    labels_folder = DATASET_PATH / split / "labels"
    images_folder = DATASET_PATH / split / "images"

    pothole_images = []

    if not labels_folder.exists():
        print(f"\nLabels folder not found: {labels_folder}")
        return pothole_images

    for label_file in labels_folder.glob("*.txt"):
        try:
            content = label_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            content = label_file.read_text().splitlines()

        has_pothole = any(
            line.strip().split()[0] == POTHOLE_CLASS_ID
            for line in content
            if line.strip() and len(line.strip().split()) >= 5
        )

        if has_pothole:
            image_found = False

            for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                image_path = images_folder / (label_file.stem + extension)

                if image_path.exists():
                    pothole_images.append(image_path)
                    image_found = True
                    break

            if not image_found:
                print(f"Image not found for label: {label_file.name}")

    return pothole_images


print("=" * 60)
print("PAVE SPOT - RDD2022 POTHOLE CHECK")
print("=" * 60)

all_pothole_images = []

for split in ["train", "val", "test"]:
    images = find_pothole_images(split)

    print(f"\n{split.upper()}")
    print(f"Pothole images: {len(images)}")

    all_pothole_images.extend(images)

print("\n" + "=" * 60)
print(f"TOTAL POTHOLE IMAGES: {len(all_pothole_images)}")
print("=" * 60)

if len(all_pothole_images) >= 1000:
    print("\nSUCCESS: We have enough images for our 1,000-image dataset.")
else:
    print("\nWARNING: We have fewer than 1,000 pothole images.")
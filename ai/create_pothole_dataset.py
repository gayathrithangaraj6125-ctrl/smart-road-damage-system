from pathlib import Path
import random
import shutil

# ============================================================
# PAVE SPOT - CREATE 1,000 IMAGE POTHOLE DATASET
# ============================================================

# Original RDD2022 dataset
SOURCE = Path(r"C:\Users\stgrj\Downloads\archive (7)")

# New dataset location inside the project
DESTINATION = Path(r"D:\smart-road-damage-system\ai\pothole_dataset")

# Original pothole class ID
POTHOLE_CLASS_ID = "3"

# New dataset class ID
NEW_CLASS_ID = "0"

# Required number of images
TOTAL_IMAGES = 1000

# Dataset split
TRAIN_COUNT = 700
VAL_COUNT = 200
TEST_COUNT = 100

# Fixed random seed so the same selection can be reproduced
random.seed(42)


def find_pothole_images(split):
    """Find images whose YOLO labels contain a pothole."""

    labels_folder = SOURCE / split / "labels"
    images_folder = SOURCE / split / "images"

    pothole_images = []

    if not labels_folder.exists():
        print(f"Labels folder not found: {labels_folder}")
        return pothole_images

    for label_file in labels_folder.glob("*.txt"):

        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label_file.read_text().splitlines()

        # Check whether this label file contains class 3 = pothole
        contains_pothole = False

        for line in lines:
            parts = line.strip().split()

            if len(parts) >= 5 and parts[0] == POTHOLE_CLASS_ID:
                contains_pothole = True
                break

        if not contains_pothole:
            continue

        # Find corresponding image
        for extension in [".jpg", ".jpeg", ".png",
                          ".JPG", ".JPEG", ".PNG"]:

            image_file = images_folder / (label_file.stem + extension)

            if image_file.exists():

                pothole_images.append(
                    (image_file, label_file)
                )

                break

    return pothole_images


def create_folders():

    folders = [
        DESTINATION / "train" / "images",
        DESTINATION / "train" / "labels",
        DESTINATION / "val" / "images",
        DESTINATION / "val" / "labels",
        DESTINATION / "test" / "images",
        DESTINATION / "test" / "labels",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def create_label(source_label, destination_label):

    """Keep only pothole annotations and change class 3 → 0."""

    new_lines = []

    try:
        lines = source_label.read_text(
            encoding="utf-8"
        ).splitlines()
    except UnicodeDecodeError:
        lines = source_label.read_text().splitlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) >= 5 and parts[0] == POTHOLE_CLASS_ID:

            # Change class ID 3 → 0
            parts[0] = NEW_CLASS_ID

            new_lines.append(" ".join(parts))

    destination_label.write_text(
        "\n".join(new_lines),
        encoding="utf-8"
    )


def copy_dataset(items, split):

    image_folder = DESTINATION / split / "images"
    label_folder = DESTINATION / split / "labels"

    for image_file, label_file in items:

        # Copy image
        shutil.copy2(
            image_file,
            image_folder / image_file.name
        )

        # Create pothole-only label
        create_label(
            label_file,
            label_folder / label_file.name
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 65)
print("PAVE SPOT - CREATING POTHOLE DATASET")
print("=" * 65)

print("\nSearching for pothole images...")

all_images = []

for split in ["train", "val", "test"]:

    images = find_pothole_images(split)

    print(f"{split.upper():5} : {len(images)} pothole images")

    all_images.extend(images)

print("\nTotal available pothole images:", len(all_images))

# Check whether enough images exist
if len(all_images) < TOTAL_IMAGES:

    print("\nERROR: Not enough pothole images.")
    print("Required:", TOTAL_IMAGES)
    print("Available:", len(all_images))

    raise SystemExit


# Remove duplicate image paths if any
unique_images = {}

for image_file, label_file in all_images:
    unique_images[str(image_file)] = (
        image_file,
        label_file
    )

all_images = list(unique_images.values())

print("Unique pothole images:", len(all_images))


# Randomly select exactly 1,000 images
random.shuffle(all_images)

selected = all_images[:TOTAL_IMAGES]

train_items = selected[:TRAIN_COUNT]
val_items = selected[
    TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
]
test_items = selected[
    TRAIN_COUNT + VAL_COUNT:
]


print("\nSelected dataset:")
print("Train:", len(train_items))
print("Val  :", len(val_items))
print("Test :", len(test_items))
print("Total:", len(selected))


# Create destination folders
print("\nCreating folders...")

create_folders()


# Copy files
print("\nCopying TRAIN files...")
copy_dataset(train_items, "train")

print("Copying VAL files...")
copy_dataset(val_items, "val")

print("Copying TEST files...")
copy_dataset(test_items, "test")


# Create new data.yaml
yaml_content = """# PAVE SPOT - Pothole Detection Dataset

path: .

train: train/images
val: val/images
test: test/images

nc: 1
names: ['Pothole']
"""

yaml_file = DESTINATION / "data.yaml"

yaml_file.write_text(
    yaml_content,
    encoding="utf-8"
)


print("\n" + "=" * 65)
print("DATASET CREATION COMPLETE")
print("=" * 65)

print("\nDataset location:")
print(DESTINATION)

print("\nFinal dataset:")
print("Train:", len(train_items), "images")
print("Val  :", len(val_items), "images")
print("Test :", len(test_items), "images")
print("Total:", len(selected), "images")

print("\nClass:")
print("0 = Pothole")

print("\ndata.yaml created successfully.")

print("\nModule 1 dataset preparation is almost complete!")
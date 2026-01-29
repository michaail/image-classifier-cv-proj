import requests
import zipfile
import os
import shutil
import random
from glob import glob

def load_dataset(url: str, extract_dir: str = "dataset"):
    """Pobiera dataset.zip spod `url`, rozpakowuje i zwraca ścieżki do splitów.

    Zwróć np. dict z kluczami:
    - real_train, real_val, real_test
    - synthetic_train, synthetic_val, synthetic_test
    - full_train, full_val, full_test
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open("dataset.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # extract zip file to a folder
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    reorganize_and_split_dataset(os.path.join(extract_dir, "bike-classifier-dataset-main"))

    # return paths to train/val/test folders
    return

def reorganize_and_split_dataset(source_base_path, target_base_path='dataset_split'):
    """
    Splits the dataset into real/synthetic/full with train/val/test structure.
    Counts: Train: 10, Val: 3, Test: 2 (per class, per subset).
    """

    # Where the unzipped data lives currently
    source_real = os.path.join(source_base_path, 'real')
    source_syn  = os.path.join(source_base_path, 'synthetic')

    # Define split counts
    SPLIT_COUNTS = {'train': 10, 'val': 3, 'test': 2}

    # 2. cleanup target if exists (to avoid duplicates on re-run)
    if os.path.exists(target_base_path):
        shutil.rmtree(target_base_path)

    # 3. Detect Classes (folders like 'class 1', 'class 2')
    classes = [d for d in os.listdir(source_real) if os.path.isdir(os.path.join(source_real, d))]
    print(f"ℹ️  Detected classes: {classes}")

    # Helper function to copy files
    def copy_files(file_list, subset_name, split_name, class_name):
        dest_dir = os.path.join(target_base_path, subset_name, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for file_path in file_list:
            shutil.copy2(file_path, dest_dir)

    # 4. Processing Loop
    for class_name in classes:
        print(f"🔄 Processing: {class_name}...")

        # Get all images for this class
        real_imgs = glob(os.path.join(source_real, class_name, '*.*'))
        syn_imgs = glob(os.path.join(source_syn, class_name, '*.*'))

        # Shuffle for randomness (reproducible)
        random.seed(42)
        random.shuffle(real_imgs)
        random.shuffle(syn_imgs)

        # Verify we have enough images
        required = sum(SPLIT_COUNTS.values()) # 15
        if len(real_imgs) < required or len(syn_imgs) < required:
            print(f"⚠️  Warning: Not enough images in {class_name}. Need {required}.")
            continue

        # --- PROCESS REAL ---
        # Slice the list: 0-10, 10-13, 13-15
        r_train = real_imgs[:10]
        r_val   = real_imgs[10:13]
        r_test  = real_imgs[13:15]

        copy_files(r_train, 'real', 'train', class_name)
        copy_files(r_val,   'real', 'val',   class_name)
        copy_files(r_test,  'real', 'test',  class_name)

        # --- PROCESS SYNTHETIC ---
        s_train = syn_imgs[:10]
        s_val   = syn_imgs[10:13]
        s_test  = syn_imgs[13:15]

        copy_files(s_train, 'synthetic', 'train', class_name)
        copy_files(s_val,   'synthetic', 'val',   class_name)
        copy_files(s_test,  'synthetic', 'test',  class_name)

        # --- PROCESS FULL (Real + Synthetic combined) ---
        # We combine the splits we just created to ensure consistency
        copy_files(r_train + s_train, 'full', 'train', class_name)
        copy_files(r_val   + s_val,   'full', 'val',   class_name)
        copy_files(r_test  + s_test,  'full', 'test',  class_name)

    print(f"\n✅ Done! Dataset organized at: {target_base_path}")
    return target_base_path


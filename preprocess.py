# preprocess.py: Recursive 3D to 2D Slices Preprocessing & Normalization
import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

DATASET_DIR = r"D:\MDM_Project\Dataset"
PROCESSED_DIR = "./processed_data"
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET_SIZE = (128, 128)

def preprocess_volumes():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return
        
    # Recursively find all patient directories by looking for folders containing '_flair.nii.gz'
    print("Searching for patient folders recursively...")
    flair_files = glob.glob(os.path.join(DATASET_DIR, "**", "*_flair.nii.gz"), recursive=True)
    
    if not flair_files:
        print(f"Error: No BraTS files found inside {DATASET_DIR}. Please check your dataset path.")
        return
        
    # Extract unique patient directories from the found flair files
    patient_dirs = list(set([os.path.dirname(f) for f in flair_files]))
    print(f"Found {len(patient_dirs)} patient folders!")
    
    slice_counter = 0
    
    for p_dir in patient_dirs:
        p_name = os.path.basename(p_dir)
        
        flair_matches = glob.glob(os.path.join(p_dir, "*_flair.nii.gz"))
        t1_matches = glob.glob(os.path.join(p_dir, "*_t1.nii.gz"))
        t1ce_matches = glob.glob(os.path.join(p_dir, "*_t1ce.nii.gz"))
        t2_matches = glob.glob(os.path.join(p_dir, "*_t2.nii.gz"))
        seg_matches = glob.glob(os.path.join(p_dir, "*_seg.nii.gz"))
        
        if not (flair_matches and t1_matches and t1ce_matches and t2_matches and seg_matches):
            print(f"Skipping {p_name}: Missing one or more modality files.")
            continue
            
        # Load volumes
        flair = nib.load(flair_matches[0]).get_fdata()
        t1 = nib.load(t1_matches[0]).get_fdata()
        t1ce = nib.load(t1ce_matches[0]).get_fdata()
        t2 = nib.load(t2_matches[0]).get_fdata()
        seg = nib.load(seg_matches[0]).get_fdata().astype(np.uint8)
        
        # Remap BraTS labels: 4 -> 3
        seg[seg == 4] = 3
        
        # Normalize modalities (z-score over non-zero brain voxels)
        mods = [flair, t1, t1ce, t2]
        norm_mods = []
        for m in mods:
            brain_mask = m > 0
            if brain_mask.sum() > 0:
                mean = m[brain_mask].mean()
                std = m[brain_mask].std()
                m_norm = np.zeros_like(m)
                m_norm[brain_mask] = (m[brain_mask] - mean) / (std + 1e-8)
            else:
                m_norm = m
            norm_mods.append(m_norm)
            
        # Extract axial slices
        depth = seg.shape[2]
        for z in range(depth):
            seg_slice = seg[:, :, z]
            
            # Keep all tumor slices + random sample of non-tumor slices (20%)
            has_tumor = np.sum(seg_slice > 0) > 0
            if not has_tumor and np.random.rand() > 0.2:
                continue
                
            slice_stack = np.stack([m[:, :, z] for m in norm_mods], axis=0).astype(np.float32)
            h, w = slice_stack.shape[1], slice_stack.shape[2]
            
            def resize_or_crop(img_2d):
                out = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1]), dtype=img_2d.dtype)
                start_h = max(0, (h - TARGET_SIZE[0]) // 2)
                start_w = max(0, (w - TARGET_SIZE[1]) // 2)
                end_h = min(h, start_h + TARGET_SIZE[0])
                end_w = min(w, start_w + TARGET_SIZE[1])
                crop = img_2d[start_h:end_h, start_w:end_w]
                ch, cw = crop.shape
                out[:ch, :cw] = crop
                return out

            processed_stack = np.stack([resize_or_crop(slice_stack[c]) for c in range(4)], axis=0)
            processed_seg = resize_or_crop(seg_slice)
            
            np.save(os.path.join(PROCESSED_DIR, f"{p_name}_z{z:03d}_img.npy"), processed_stack)
            np.save(os.path.join(PROCESSED_DIR, f"{p_name}_z{z:03d}_seg.npy"), processed_seg)
            slice_counter += 1

    print(f"Preprocessing complete! Saved {slice_counter} 2D slices to '{PROCESSED_DIR}'.")

if __name__ == "__main__":
    preprocess_volumes()
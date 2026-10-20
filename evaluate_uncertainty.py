# evaluate_uncertainty.py: MC Dropout, Uncertainty Maps, and RPS Weight Tuning
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import minimize

from dataset_loader import DATASET_DIR, Brats2DSliceDataset, get_patient_splits, ResUNetWithFiLM

CHECKPOINT_DIR = "./brats_project/checkpoints"
RESULTS_DIR = "./brats_project/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def mc_dropout_inference(model, img, mask, num_passes=10):
    model.train() # Keep dropout active during inference
    preds = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(img, mask)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            
    preds = np.stack(preds, axis=0) # Shape: (num_passes, B, C, H, W)
    mean_preds = np.mean(preds, axis=0) # Shape: (B, C, H, W)
    
    # Predictive Entropy as uncertainty measure: H(p) = -sum(p * log(p))
    eps = 1e-8
    entropy = -np.sum(mean_preds * np.log(mean_preds + eps), axis=1) # Shape: (B, H, W)
    
    return mean_preds, entropy

def compute_boundary_uncertainty(entropy_map, pred_mask):
    # Detect boundaries using simple morphological gradient or edge detection on prediction
    # Here we use gradient magnitude on the predicted class map
    from scipy.ndimage import morphological_gradient
    b_map = np.zeros_like(entropy_map)
    for i in range(pred_mask.shape[0]):
        grad = morphological_gradient(pred_mask[i], size=(3, 3)) > 0
        b_map[i] = entropy_map[i] * grad
    return b_map

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    _, val_data, test_data = get_patient_splits(DATASET_DIR)
    val_dataset = Brats2DSliceDataset(val_data[0], val_data[1], is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = ResUNetWithFiLM(in_channels=4, out_classes=4, dropout_p=0.2).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Please complete training first.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    print("Running MC Dropout inference on validation set...")
    all_mean_uncertainties = []
    all_boundary_uncertainties = []
    all_tumor_areas = []
    all_missing_penalties = []
    all_errors = []
    
    model.eval()
    for imgs, segs, masks in tqdm(val_loader, desc="MC Dropout Val"):
        imgs, segs, masks = imgs.to(device), segs.to(device), masks.to(device)
        
        # Perform MC Dropout (10 passes)
        mean_preds, entropy = mc_dropout_inference(model, imgs, masks, num_passes=10)
        pred_labels = np.argmax(mean_preds, axis=1) # (B, H, W)
        
        # Compute error map (1 where prediction != ground truth, else 0)
        seg_np = segs.cpu().numpy()
        error_map = (pred_labels != seg_np).astype(np.float32)
        
        boundary_unc = compute_boundary_uncertainty(entropy, pred_labels)
        
        for b in range(imgs.shape[0]):
            mean_unc = np.mean(entropy[b])
            b_unc = np.mean(boundary_unc[b])
            tumor_area = np.sum(pred_labels[b] > 0) / (128 * 128)
            missing_penalty = 1.0 - (masks[b].sum().item() / 4.0)
            slice_error = np.mean(error_map[b])
            
            all_mean_uncertainties.append(mean_unc)
            all_boundary_uncertainties.append(b_unc)
            all_tumor_areas.append(tumor_area)
            all_missing_penalties.append(missing_penalty)
            all_errors.append(slice_error)
            
    # Normalize features for RPS calculation
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
    u_mean = normalize(all_mean_uncertainties)
    u_bound = normalize(all_boundary_uncertainties)
    t_area = normalize(all_tumor_areas)
    m_pen = normalize(all_missing_penalties)
    errors = np.array(all_errors)
    
    # Tune Review Priority Score (RPS) weights: w1*u_mean + w2*u_bound + w3*t_area + w4*m_pen
    # Objective: Maximize correlation between RPS and actual slice error
    def objective(weights):
        w1, w2, w3, w4 = weights
        rps = w1 * u_mean + w2 * u_bound + w3 * t_area + w4 * m_pen
        # Negative correlation because optimizer minimizes
        corr = np.corrcoef(rps, errors)[0, 1]
        return -corr if not np.isnan(corr) else 0.0
        
    res = minimize(objective, [0.4, 0.3, 0.2, 0.1], bounds=[(0,1)]*4, method='Nelder-Mead')
    best_w = res.x
    best_w = best_w / np.sum(best_w) # normalize weights to sum to 1
    
    print("\n--- Review Priority Score (RPS) Weights Tuned ---")
    print(f"w1 (Mean Uncertainty): {best_w[0]:.4f}")
    print(f"w2 (Boundary Uncertainty): {best_w[1]:.4f}")
    print(f"w3 (Normalized Tumor Area): {best_w[2]:.4f}")
    print(f"w4 (Missing Modality Penalty): {best_w[3]:.4f}")
    
    # Save weights
    np.save(os.path.join(RESULTS_DIR, "best_rps_weights.npy"), best_w)
    print("Stage 6 complete! RPS weights saved.")

if __name__ == "__main__":
    run_evaluation()
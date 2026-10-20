# evaluate_test_triage.py: Missing Modalities Evaluation & Triage Curve Generation
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from dataset_loader import DATASET_DIR, Brats2DSliceDataset, get_patient_splits, ResUNetWithFiLM

CHECKPOINT_DIR = "./brats_project/checkpoints"
RESULTS_DIR = "./brats_project/results"
PLOTS_DIR = "./brats_project/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def compute_dice(pred, target, num_classes=4, smooth=1e-5):
    dices = []
    for c in range(num_classes):
        p_c = (pred == c).astype(np.float32)
        t_c = (target == c).astype(np.float32)
        intersection = np.sum(p_c * t_c)
        union = np.sum(p_c) + np.sum(t_c)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dices.append(dice)
    return dices

def run_test_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating test set on device: {device}")
    
    _, _, test_data = get_patient_splits(DATASET_DIR)
    test_dataset = Brats2DSliceDataset(test_data[0], test_data[1], is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = ResUNetWithFiLM(in_channels=4, out_classes=4, dropout_p=0.2).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Please complete training first.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load tuned RPS weights
    weights_path = os.path.join(RESULTS_DIR, "best_rps_weights.npy")
    w1, w2, w3, w4 = np.load(weights_path) if os.path.exists(weights_path) else [0.4, 0.3, 0.2, 0.1]
    
    # Missing modality scenarios: 4-bit masks [FLAIR, T1, T1ce, T2]
    scenarios = {
        "Full Modalities": [1.0, 1.0, 1.0, 1.0],
        "Only T1ce Missing": [1.0, 1.0, 0.0, 1.0],
        "Only FLAIR Missing": [0.0, 1.0, 1.0, 1.0],
        "Only T2 Missing": [1.0, 1.0, 1.0, 0.0],
        "Only T1 Missing": [1.0, 0.0, 1.0, 1.0],
        "T1ce + FLAIR Missing": [0.0, 1.0, 0.0, 1.0]
    }
    
    dice_results = {scen: [] for scen in scenarios.keys()}
    
    print("\n--- Evaluating Dice Scores under Missing Modalities ---")
    with torch.no_grad():
        for imgs, segs, _ in tqdm(test_loader, desc="Test Scenarios"):
            imgs, segs = imgs.to(device), segs.to(device)
            segs_np = segs.cpu().numpy()
            
            for scen_name, mask_vals in scenarios.items():
                mask_tensor = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(0).repeat(imgs.shape[0], 1).to(device)
                
                # Apply missing modalities to input images
                masked_imgs = imgs.clone()
                for i in range(4):
                    if mask_vals[i] == 0.0:
                        masked_imgs[:, i, :, :] = 0.0
                        
                logits = model(masked_imgs, mask_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                for b in range(imgs.shape[0]):
                    d = compute_dice(preds[b], segs_np[b])
                    dice_results[scen_name].append(np.mean(d))
                    
    # Summarize Dice table
    summary_table = {scen: np.mean(scores) for scen, scores in dice_results.items()}
    df_dice = pd.DataFrame(list(summary_table.items()), columns=["Scenario", "Mean Dice"])
    print("\n", df_dice)
    df_dice.to_csv(os.path.join(RESULTS_DIR, "missing_modalities_dice.csv"), index=False)
    
    # Triage Curve & Uncertainty Evaluation on Full Modality Test Set
    print("\n--- Computing Triage Curve and Review Priority Scores ---")
    rps_scores, actual_errors, tumor_areas_list = [], [], []
    
    with torch.no_grad():
        for imgs, segs, _ in tqdm(test_loader, desc="Triage Evaluation"):
            imgs, segs = imgs.to(device), segs.to(device)
            mask_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32).unsqueeze(0).repeat(imgs.shape[0], 1).to(device)
            
            # MC Dropout for uncertainty
            model.train()
            mc_preds = []
            for _ in range(5):
                logits = model(imgs, mask_tensor)
                mc_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            model.eval()
            
            mc_preds = np.stack(mc_preds, axis=0)
            mean_p = np.mean(mc_preds, axis=0)
            entropy = -np.sum(mean_p * np.log(mean_p + 1e-8), axis=1)
            
            pred_labels = np.argmax(mean_p, axis=1)
            segs_np = segs.cpu().numpy()
            error_map = (pred_labels != segs_np).astype(np.float32)
            
            for b in range(imgs.shape[0]):
                mean_unc = np.mean(entropy[b])
                b_unc = mean_unc * 0.5
                t_area = np.sum(pred_labels[b] > 0) / (128 * 128)
                m_pen = 0.0
                
                rps = w1 * mean_unc + w2 * b_unc + w3 * t_area + w4 * m_pen
                rps_scores.append(rps)
                actual_errors.append(np.mean(error_map[b]) > 0.05)
                tumor_areas_list.append(t_area)
                
    rps_scores = np.array(rps_scores)
    actual_errors = np.array(actual_errors)
    tumor_areas = np.array(tumor_areas_list)
    
    def get_triage_curve(scores, errors):
        sorted_indices = np.argsort(scores)[::-1]
        sorted_errors = errors[sorted_indices]
        total_errors = np.sum(sorted_errors)
        if total_errors == 0:
            return np.zeros(10), np.zeros(10)
        
        percentages = np.linspace(0.01, 1.0, 100)
        captured = []
        for p in percentages:
            k = int(p * len(scores))
            captured.append(np.sum(sorted_errors[:k]) / total_errors * 100)
        return percentages * 100, captured
        
    pct_reviewed, rps_captured = get_triage_curve(rps_scores, actual_errors)
    _, random_captured = get_triage_curve(np.random.permutation(len(rps_scores)), actual_errors)
    _, tumor_captured = get_triage_curve(tumor_areas, actual_errors)
    
    idx_20 = min(20, len(pct_reviewed)-1)
    print(f"\n--- Triage Results at 20% Slices Reviewed ---")
    print(f"RPS Strategy - Errors Captured: {rps_captured[idx_20]:.2f}%")
    print(f"Tumor Area Only - Errors Captured: {tumor_captured[idx_20]:.2f}%")
    print(f"Random Review - Errors Captured: {random_captured[idx_20]:.2f}%")
    
    # Plot Triage Curve
    plt.figure(figsize=(8, 6))
    plt.plot(pct_reviewed, rps_captured, label="Review Priority Score (Ours)", linewidth=2.5, color='blue')
    plt.plot(pct_reviewed, tumor_captured, label="Tumor Area Only", linestyle='--', color='orange')
    plt.plot(pct_reviewed, random_captured, label="Random Order", linestyle=':', color='gray')
    plt.xlabel("Percentage of Slices Reviewed (%)", fontsize=12)
    plt.ylabel("Percentage of Errors Captured (%)", fontsize=12)
    plt.title("Radiologist Review Triage Curve", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "triage_curve.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Triage curve plot saved to {plot_path}")
    print("Stage 7 & 8 evaluation complete!")

if __name__ == "__main__":
    run_test_evaluation()
    
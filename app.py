# Neutralize PyTorch class path inspection for Streamlit compatibility
import torch
try:
    torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Triage & Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Color Map for Tumor Classes (BG=Black, NC=Red, ED=Green, ET=Yellow)
LABEL_CMAP = ListedColormap([(0, 0, 0), (1, 0, 0), (0, 0.8, 0), (1, 1, 0)])

@st.cache_resource
def load_model_and_data():
    from dataset_loader import ResUNetWithFiLM, DATASET_DIR, Brats2DSliceDataset, get_patient_splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "./brats_project/checkpoints/best_model.pth"
    
    model = ResUNetWithFiLM(4, 4, 0.2).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    try:
        _, _, (test_imgs, test_segs) = get_patient_splits(DATASET_DIR)
        test_dataset = Brats2DSliceDataset(test_imgs, test_segs, is_training=False)
    except Exception:
        test_dataset = None
        
    return model, device, test_dataset

model, device, test_dataset = load_model_and_data()

# Sidebar Controls
st.sidebar.header("1. Clinical Input & Protocol")
input_method = st.sidebar.radio("Choose Input Source", ["Upload .npy File", "Load Test Dataset Sample"])

img_tensor = None
gt_mask_viz = None

if input_method == "Upload .npy File":
    uploaded_file = st.sidebar.file_uploader("Upload Axial Slice (.npy)", type=["npy"])
    if uploaded_file is not None:
        try:
            data = np.load(uploaded_file)
            st.sidebar.write(f"Uploaded array shape: {data.shape}")
            
            if data.ndim == 3 and data.shape[0] == 4:
                img_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            elif data.ndim == 3 and data.shape[2] == 4:
                data_t = np.transpose(data, (2, 0, 1))
                img_tensor = torch.tensor(data_t, dtype=torch.float32).unsqueeze(0).to(device)
            elif data.ndim == 2:
                st.sidebar.warning("Uploaded 2D array detected. Replicating across 4 modalities.")
                data_4ch = np.stack([data]*4, axis=0)
                img_tensor = torch.tensor(data_4ch, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                img_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
        except Exception as e:
            st.sidebar.error(f"Error loading numpy file: {e}")
else:
    if test_dataset is not None and len(test_dataset) > 0:
        sample_idx = st.sidebar.slider("Select Test Slice Index", 0, min(100, len(test_dataset)-1), 10)
        img, seg, msk = test_dataset[sample_idx]
        img_tensor = img.unsqueeze(0).to(device)
        gt_mask_viz = seg.numpy()
        st.sidebar.success(f"Loaded test sample index {sample_idx}")
    else:
        st.sidebar.warning("Dataset path not found. Please check your dataset directory.")

st.sidebar.markdown("---")
st.sidebar.subheader("Modality Availability Mask")
has_flair = st.sidebar.checkbox("FLAIR", value=True)
has_t1 = st.sidebar.checkbox("T1-weighted", value=True)
has_t1ce = st.sidebar.checkbox("T1-contrast enhanced (T1ce)", value=True)
has_t2 = st.sidebar.checkbox("T2-weighted", value=True)

availability_mask = torch.tensor([[int(has_flair), int(has_t1), int(has_t1ce), int(has_t2)]], dtype=torch.float32).to(device)

# Main Dashboard Title
st.title("🧠 Uncertainty-Guided Slice Triage System")
st.markdown("*Real-time multimodal brain tumor segmentation and epistemic uncertainty triage under missing MRI sequences.*")
st.markdown("---")

def mc_dropout_inference(img_tensor, mask_tensor, passes=10):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()
            
    with torch.no_grad():
        preds = []
        for _ in range(passes):
            logits = model(img_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            
    mean_p = np.mean(preds, axis=0)
    entropy = -np.sum(mean_p * np.log(mean_p + 1e-8), axis=1)
    return mean_p, entropy

if img_tensor is not None:
    try:
        for idx, active in enumerate([has_flair, has_t1, has_t1ce, has_t2]):
            if not active and img_tensor.shape[1] > idx:
                img_tensor[:, idx, :, :] = 0.0

        with st.spinner("Running 2D ResUNet inference & MC Dropout stochastic passes..."):
            mean_probs, uncertainty_map = mc_dropout_inference(img_tensor, availability_mask)
            pred_mask = mean_probs[0].argmax(0)
            mean_entropy_val = float(uncertainty_map[0].mean())
            rps_score = min(0.98, max(0.12, mean_entropy_val * 2.5 + (0.3 if not has_t1ce else 0.0)))

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Review Priority Score (RPS)", f"{rps_score:.3f}", "High Urgency" if rps_score > 0.5 else "Low Urgency")
        col_m2.metric("Mean Epistemic Uncertainty", f"{mean_entropy_val:.4f}", "MC Entropy")
        col_m3.metric("Protocol Status", "Complete" if all([has_flair, has_t1, has_t1ce, has_t2]) else "Incomplete (FiLM Adapted)")

        st.markdown("### Visual Inference Results")
        
        c1, c2, c3, c4 = st.columns(4)
        t1ce_idx = 2 if img_tensor.shape[1] > 2 else 0
        t1ce_slice = img_tensor[0, t1ce_idx].cpu().numpy()
        
        with c1:
            st.subheader("Input Scan (T1ce)")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.rot90(t1ce_slice), cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
            
        with c2:
            st.subheader("Ground Truth / Ref")
            fig, ax = plt.subplots(figsize=(4, 4))
            if gt_mask_viz is not None:
                ax.imshow(np.rot90(gt_mask_viz), cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
            else:
                ax.imshow(availability_mask.cpu().numpy().reshape(1, 4), cmap="coolwarm", vmin=0, vmax=1, aspect='auto')
                ax.set_title("Modality Mask Active", fontsize=10)
            ax.axis("off")
            st.pyplot(fig)
            
        with c3:
            st.subheader("ResUNet Prediction")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.rot90(pred_mask), cmap=LABEL_CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax.axis("off")
            st.pyplot(fig)
            
        with c4:
            st.subheader("Uncertainty Map")
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(np.rot90(uncertainty_map[0]), cmap="inferno")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Segmentation Legend:** 🔴 **Necrotic Core (NC)** | 🟢 **Peritumoral Edema (ED)** | 🟡 **Enhancing Tumor (ET)**")

    except Exception as e:
        st.error(f"Inference error occurred during execution: {e}")
        st.info("💡 Try selecting **Load Test Dataset Sample** from the sidebar instead.")

else:
    st.markdown("""
    ### Welcome to the Triage Web Interface!
    To test the system in real-time:
    1. Make sure your trained model checkpoint is located at `./brats_project/checkpoints/best_model.pth`.
    2. Choose **Load Test Dataset Sample** or **Upload .npy File** from the left sidebar.
    3. Toggle modality availability checkboxes to test missing-sequence robustness with **FiLM conditioning**.
    """)
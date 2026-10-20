# Uncertainty-Guided Slice Triage for Brain Tumor Segmentation under Missing MRI Modalities

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)

Official repository for my B.Tech Biomedical Engineering Final-Year Thesis. This project introduces an uncertainty-guided clinical triage framework using a 2D ResUNet with **FiLM conditioning** for robust brain tumor segmentation under incomplete/missing MRI sequences.

## 🧠 Key Features
- **Multimodal Adaptability**: Handles missing MRI modalities (FLAIR, T1, T1ce, T2) dynamically using Feature-wise Linear Modulation (FiLM).
- **Epistemic Uncertainty Estimation**: Utilizes Monte Carlo (MC Dropout) during inference to compute predictive entropy.
- **Automated Slice Triage**: Converts pixel uncertainty into a **Review Priority Score (RPS)** to flag ambiguous slices for radiologists.
- **Interactive Streamlit Web App**: A real-time clinical deployment dashboard for live visualization and multi-class segmentation (NC, ED, ET).

## 🚀 Quick Start (Running the Web App)
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/your-repo-name.git](https://github.com/YOUR-USERNAME/your-repo-name.git)
   cd your-repo-name
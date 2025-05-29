
# Immediate Remaining Capacity Estimation of Second-Life Batteries

This repository contains the official implementation of the paper:

**"Immediate remaining capacity estimation of heterogeneous second-life lithium-ion batteries via deep generative transfer learning"**

## 🚀 Motivation

The reuse of second-life lithium-ion batteries offers substantial sustainability and economic benefits, especially in regions with weak grid infrastructure. However, challenges arise due to:

- Lack of historical usage data.
- Heterogeneity across battery formats and degradation pathways.

This project proposes a **deep generative transfer learning** framework to:
- Generate synthetic pulse test voltage data across State-of-Charge (SOC) levels using a **Variational Autoencoder (VAE)**.
- Predict SOC from voltage response via an **SOC net**.
- Estimate Relative Remaining Capacity (RRC) using a **regression net**.
- Perform cross-domain adaptation with **Deep CORAL** to generalize across battery formats.

## 🧠 Model Architecture

The core architecture includes:
1. **VAE net**: Generates voltage response features across SOC levels.
2. **SOC net**: Predicts SOC based on voltage features.
3. **Regression net**: Estimates RRC from voltage and SOC inputs.
4. **CORAL net**: Performs domain adaptation across different battery types and degradation behaviors.

<p align="center">
  <img src="https://github.com/terencetaothucb/Immediate-remaining-capacity-estiamtion-of-second-life-batteries/blob/main/Structure.png" width="600"/>
</p>

---

## 📁 Repository Structure

```bash
Immediate-remaining-capacity-estimation-of-second-life-batteries/
│
├── DeepCORAL.py                        # Deep CORAL domain adaptation and training pipeline
├── DeepCORAL-Results-to-Excel.py      # Output results processing and Excel export
├── Benckmarking-Models.py             # Comparison with other ML models (e.g., SVM, GPR)
├── Model-Performance-Comparison.py    # Visualization of model performances across domains
├── Risk-Analysis.py                   # Analysis of prediction uncertainty and error distributions
├── Target-Data-Availability-Analysis.py  # Sensitivity to field data availability
├── README.md                          # Project introduction and usage instructions (this file)
```

---

## 🔧 How to Use

### Set up Environment

```bash
pip install -r requirements.txt
```

> Python 3.8.15 is recommended.

## Access
Correspondence to [Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](mailto:xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](mailto:guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
## Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) at Tsinghua Berkeley Shenzhen Institute prepared the data, designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials. [MIT](https://github.com/terencetaothucb/Immediate-remaining-capacity-estiamtion-of-second-life-batteries/blob/main/LICENSE) license applied.




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

### 1. Set up Environment

```bash
pip install -r requirements.txt
```

> Python 3.8.15 is recommended.

### 2. Run Domain Adaptation Model

```bash
python DeepCORAL.py
```

### 3. Evaluate Results

```bash
python DeepCORAL-Results-to-Excel.py
```

### 4. Run Benchmarks

```bash
python Benckmarking-Models.py
```

### 5. Visualization and Analysis

```bash
python Model-Performance-Comparison.py
python Risk-Analysis.py
python Target-Data-Availability-Analysis.py
```

---

## 📄 Citation

If you find this repository helpful, please cite our paper:

Tao, S., et al. *Immediate remaining capacity estimation of heterogeneous second-life lithium-ion batteries via deep generative transfer learning*, 2025.

---

## 🤝 Contributors

- Shengyu Tao
- Xuan Zhang
- Scott Moura
- Jinpeng Tian
- Guangmin Zhou

---

## 📬 Contact

For questions or collaboration, please reach out to:

- Shengyu Tao (terencetaothucb@berkeley.edu)
- Prof. Xuan Zhang (xuanzhang@sz.tsinghua.edu.cn)

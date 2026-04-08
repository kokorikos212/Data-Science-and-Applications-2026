# GaiaStellarNet: Multi-Modal Stellar Analysis

This project implements a data-driven pipeline for analyzing **Gaia mission data**. It combines custom streaming architectures, statistical learning, and robust validation to predict stellar properties—specifically **Absolute Magnitude ($M_G$)**—using multi-modal astronomical features.

---

## 🚀 Key Capabilities

* **Streaming Data Factory:** Efficiently pulls data from Hugging Face streams, enabling the processing of massive astronomical datasets without high memory overhead.
* **Integrity-First Pipeline:** Automated filters for physical completeness, handling the "Log of Zero" parallax traps and stripping corrupted (NaN) entries.
* **Hybrid Modeling Suite:**
    * **Neural Networks:** PyTorch-based regression for complex pattern recognition.
    * **Random Forests:** High-accuracy ensemble models for tabular data.
    * **PCA Integration:** Dimensionality reduction to compress auxiliary modalities into unified feature sets.
* **Scientific Visualization:** Automated generation of **Double HR Diagrams** to compare ground truth Gaia observations against model predictions.

---

## 📂 Project Structure

| Component | Responsibility |
| :--- | :--- |
| **DataFactory** | The engine for streaming, flattening, and validating datasets. |
| **StellarModels** | Architecture definitions for PyTorch and Scikit-Learn estimators. |
| **Preprocessing** | Logic for Principal Component Analysis (PCA) and standard scaling. |
| **Evaluation** | Tools for calculating $R^2$, MSE, and generating Hertzsprung-Russell diagrams. |

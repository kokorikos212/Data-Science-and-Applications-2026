# Data-Science-and-Applications-2026
Project0:

GaiaStellarNet: Multi-Modal Stellar Analysis
This project implements a data-driven pipeline for analyzing Gaia mission data. It combines custom streaming architectures, statistical learning, and robust validation to predict stellar properties—specifically Absolute Magnitude ($M_G$)—using multi-modal astronomical features.

🚀 Key Capabilities
Streaming Data Factory: Efficiently pulls data from Hugging Face streams, enabling the processing of massive astronomical datasets without high memory overhead.
Integrity-First Pipeline: Automated filters for physical completeness, handling the "Log of Zero" parallax traps and stripping corrupted (NaN) entries.
Hybrid Modeling Suite:
Neural Networks: PyTorch-based regression for complex pattern recognition.
Random Forests: High-accuracy ensemble models for tabular data.
PCA Integration: Dimensionality reduction to compress auxiliary modalities into unified feature sets.
Scientific Visualization: Automated generation of Double HR Diagrams to compare ground truth Gaia observations against model predictions.
📂 Project Structure
Component	Responsibility
DataFactory	A multimodal engine that manages and joins astronomical datasets via streaming; handles structured DataFrame conversion and spatial cross-matches using celestial coordinates.
ExploratoryAnalyzer	Computes similarity matrices and descriptive statistics to quantify the relationships within the stellar data matrix.
FeatureEngineer	Implements likelihood scoring based on 1D probability distributions (PDFs) for stellar properties like Color and Extent.
UniverseVisualizer	A dedicated suite for analyzing the statistical distributions and composition of the multimodal universe manifest.
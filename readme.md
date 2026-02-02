
# Instrument Tuning Detection using Deep Learning

**A machine learning pipeline that detects whether an instrument is in tune or out of tune using deep learning models and audio feature extraction.**

This project was originally part of the **CS8321 Final Project** coursework and has been refactored, enhanced, and packaged into a clear, reproducible repository suitable for portfolio and professional use.

---

## 🚀 Project Overview

Musical instrument tuning detection involves analyzing audio signals to determine if an instrument is playing in tune. This project builds and evaluates deep learning models that classify audio samples as **in-tune** or **out-of-tune** based on learned representations from audio features.

---


## 🛠️ Key Features

- **Audio Feature Extraction:** Compute relevant audio features from instrument recordings to serve as input to ML models.
- **Deep Learning Classification:** Train, evaluate, and save models using popular architectures tailored for audio inspection.
- **Model Evaluation:** Analyze performance using standard metrics such as accuracy, confusion matrices, and spectrogram insights.
- **Reproducible Workflow:** Notebooks and scripts that document the end-to-end process from raw data to model evaluation.

---

## 🧠 Approach

1. **Data Preprocessing**
   - Convert raw audio files into structured datasets.
   - Extract meaningful features such as spectrograms and embeddings (e.g., VGGish) for model training.

2. **Model Training**
   - Train neural network models using extracted features.
   - Experiment with architectures such as CNNs and MLPs for tuning detection.

3. **Evaluation**
   - Evaluate trained models on holdout sets.
   - Use performance metrics and visualizations to validate model behavior.

4. **Deployment Thoughts**
   - While this version focuses on research and modeling, the pipeline is structured for easy deployment into cloud services (e.g., GCP, AWS).

---

## ⚙️ Requirements

numpy
pandas
scikit-learn
matplotlib
seaborn
librosa
soundfile
tensorflow
keras
torch
torchaudio
tqdm
jupyter
ipykernel


⏱ How to Run

To train a model:

python src/train_model.py --data-dir ./data

To evaluate:

python src/evaluate.py --model models/out_of_tune_detector_model.h5

To generate feature representations:

python src/features.py --input ./data/*.wav

(Adjust paths as needed)





📈 Visualizations

The notebooks/ folder includes:
	•	Spectrogram Analysis
	•	Feature Distribution
	•	Training and Validation Curves
	•	Confusion Matrix Visualization





This project is intended for research, education, and portfolio demonstration. When applying to professional roles, be transparent about the origin (coursework) and your extensions.


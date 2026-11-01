# SkinStackNet 🧠
A Hybrid Deep Stacking Ensemble Model for Skin Disease Image Classification

SkinStackNet is an academic team project that explores deep learning, attention mechanisms, and ensemble learning for multiclass skin disease image classification using dermoscopic images.

The project combines three pretrained convolutional neural networks  EfficientNetB0, MobileNetV3Large, and ResNet50 with Convolutional Block Attention Module (CBAM) attention and a dual-stacking meta-learning strategy.

The objective is to investigate whether combining complementary model predictions can improve classification performance compared with individual CNN models and a soft-voting baseline.

> **Project status:** Academic team project. Not published as a peer-reviewed research paper.

Key Features

- **Transfer learning:** Uses pretrained EfficientNetB0, MobileNetV3Large, and ResNet50.
- **Attention mechanism:** Integrates CBAM channel and spatial attention.
- **Data augmentation:** Uses MixUp augmentation during training.
- **Hybrid ensemble:** Combines model logits and softmax probabilities.
- **Meta-learning:** Uses Logistic Regression to learn from the combined model outputs.
- **Soft-voting baseline:** Averages prediction probabilities from the three base models for comparison.
- **Explainable AI:** Uses Grad-CAM visualizations to inspect image regions influencing predictions.
- **Web interface:** Includes a Flask-based application for image upload and prediction visualization.

## 🏗️ Model Architecture

The SkinStackNet workflow consists of the following stages:

1. **Image preprocessing:** Images are resized to 224 × 224 pixels, converted to RGB, and prepared for model input.
2. **Data preparation:** Class labels are encoded, and the dataset is divided into training, validation, and test sets.
3. **Data augmentation:** MixUp is used during training to increase sample diversity.
4. **Feature learning:** EfficientNetB0, MobileNetV3Large, and ResNet50 extract image features using transfer learning.
5. **Attention enhancement:** CBAM modules apply channel and spatial attention.
6. **Prediction extraction:** Each base model produces logits and softmax probabilities.
7. **Dual stacking:** The outputs are concatenated into a combined meta-feature representation.
8. **Feature standardization:** StandardScaler prepares the combined features for the meta-learner.
9. **Meta-classification:** Logistic Regression produces the final ensemble prediction.
10. **Evaluation and interpretation:** Model performance is assessed using classification metrics, confusion matrices, ROC analysis, and Grad-CAM.

### Architecture Diagram

<img width="1060" height="480" alt="image" src="https://github.com/user-attachments/assets/9480575f-c097-44d0-9f3b-cf6d6fbfe958" />


## 🧪 Dataset

The project presentation documents a publicly available skin disease image dataset obtained from Kaggle.

| Property | Description |
|---|---|
| Total images | 2,568 |
| Image type | Dermoscopic RGB images |
| Input resolution | 224 × 224 × 3 |
| Number of classes | 6 |
| Training split | 80% |
| Validation split | 10% |
| Test split | 10% |
| Dataset source | Kaggle |

### Classification Categories

1. Actinic Keratosis
2. Dermatofibroma
3. Nevus
4. Pigmented Benign Keratosis
5. Seborrheic Keratosis
6. Vascular Lesion

Dataset counts, class names, and split details should be checked against the actual data and code before publication.

**Dataset access:** Refer to the original dataset source and its usage terms. The dataset itself is not included in this repository unless redistribution is permitted.

## 🛠️ Technology Stack

- **Language:** Python 3.10
- **Deep learning:** TensorFlow 2.x, Keras
- **Machine learning:** Scikit-learn
- **Image processing:** OpenCV
- **Data handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Augmentation:** KerasCV
- **Web application:** Flask
- **Development environment:** Jupyter Notebook / Google Colab

## 📊 Experimental Results

The project presentation reports the following model comparison:

| Model | Reported accuracy |
|---|---:|
| MobileNetV3Large | 92% |
| EfficientNetB0 | 94% |
| ResNet50 | 94% |
| Hybrid Dual Stacking | 98% |

The presentation also reports a micro-average ROC AUC of approximately 0.999 for the hybrid dual-stacking model.

These figures are reported project results, not independently verified benchmarks. Their interpretation depends on the evaluation setup, dataset, and experimental methodology. Refer to the evaluation code and logs for the exact test configuration.

### Results Visualization

<img width="803" height="518" alt="image" src="https://github.com/user-attachments/assets/8c566fa5-4834-4352-b643-6d12c7058a73" />


### Confusion Matrix

<img width="1194" height="502" alt="image" src="https://github.com/user-attachments/assets/df79f3f1-af75-492c-b4ac-1a45fff9a317" />

<img width="1237" height="521" alt="image" src="https://github.com/user-attachments/assets/2a77b244-5a84-4664-88f9-8c6bb315442a" />

### Explainability with Grad-CAM

<img width="707" height="361" alt="image" src="https://github.com/user-attachments/assets/9f274ca2-055a-4693-b98b-14ca05215846" />

Grad-CAM visualizations are used to inspect which image regions influence the model's predictions. These heatmaps are exploratory interpretability outputs and do not establish clinical validity.

## 💻 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/........./SkinStackNet.git
cd SkinStackNet
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```
Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```
**Linux / macOS**
```bash
source .venv/bin/activate
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Prepare the dataset

Download the permitted dataset from its original source and configure the dataset path according to the project code.

### 5. Run the project

Follow the notebook or script entry point documented in the repository.

If the Flask application is included, provide its actual startup command here after verifying it against the application code.

## 👥 Team and Contributions

SkinStackNet was developed as an academic team project at Madanapalle Institute of Technology & Science.

**Project guide:** Dr. S. Satheesh Kumar

**Team members:**
- Uday Sree Sai P.
- Santhosh Naik B.
- Sasipreetham G.
- Vivek D.

Individual responsibilities should be documented accurately based on each contributor's work. This repository is intended to represent the collaborative project, not to imply sole authorship.

## 🚀 Future Improvements

- Evaluate generalization on independent datasets.
- Add reproducible training and evaluation scripts.
- Investigate class-wise performance and potential data leakage.
- Improve model interpretability and uncertainty estimation.
- Explore efficient deployment and inference optimization.

## ⚠️ Limitations and Disclaimer

SkinStackNet is an educational and experimental machine learning project. It is not a clinically validated medical device and must not be used as a substitute for professional medical evaluation.

Model performance on a particular dataset does not guarantee reliable predictions on unseen images or different patient populations.

## 📌 Publication Status

This repository documents an academic team project. The work has not been published as a peer-reviewed research paper.

Please refer to the original dataset, pretrained model, and third-party software licenses before redistributing any associated materials.

**Topics:** `deep-learning` · `skin-disease-classification` · `computer-vision` · `tensorflow` · `transfer-learning` · `ensemble-learning` · `explainable-ai`

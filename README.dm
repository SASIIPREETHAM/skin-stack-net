SkinStackNet is a Hybrid Deep Stacking Ensemble Model for Skin Disease Image Classification
SkinStackNet is an academic team project that explores deep learning, attention mechanisms, and ensemble learning for multiclass skin disease image classification using dermoscopic images.
The project combines three pretrained convolutional neural networks  EfficientNetB0, MobileNetV3Large, and ResNet50 with Convolutional Block Attention Module (CBAM) attention and a dual-stacking meta-learning strategy.
The objective is to investigate whether combining complementary model predictions can improve classification performance compared with individual CNN models and a soft-voting baseline.
 
Project status: Academic team project. Not published as a peer-reviewed research paper.

Key Features
Transfer learning: Uses pretrained EfficientNetB0, MobileNetV3Large, and ResNet50.
Attention mechanism: Integrates CBAM channel and spatial attention.
Data augmentation: Uses MixUp augmentation during training.
Hybrid ensemble: Combines model logits and softmax probabilities.
Meta-learning: Uses Logistic Regression to learn from the combined model outputs.
Soft-voting baseline: Averages prediction probabilities from the three base models for comparison.
Explainable AI: Uses Grad-CAM visualizations to inspect image regions influencing predictions.
Web interface: Includes a Flask-based application for image upload and prediction visualization.

 Model Architecture

The SkinStackNet workflow consists of the following stages:

1. Image preprocessing: Images are resized to 224 × 224 pixels, converted to RGB, and prepared for model input.
2. Data preparation: Class labels are encoded, and the dataset is divided into training, validation, and test sets.
3. Data augmentation: MixUp is used during training to increase sample diversity.
4. Feature learning: EfficientNetB0, MobileNetV3Large, and ResNet50 extract image features using transfer learning.
5. Attention enhancement: CBAM modules apply channel and spatial attention.
6. Prediction extraction: Each base model produces logits and softmax probabilities.
7. Dual stacking: The outputs are concatenated into a combined meta-feature representation.
8. Feature standardization: StandardScaler prepares the combined features for the meta-learner.
9. Meta-classification: Logistic Regression produces the final ensemble prediction.
10.Evaluation and interpretation: Model performance is assessed using classification metrics, confusion matrices, ROC analysis, and Grad-CAM.

🧪 Dataset

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

Classification Categories

1. Actinic Keratosis
2. Dermatofibroma
3. Nevus
4. Pigmented Benign Keratosis
5. Seborrheic Keratosis
6. Vascular Lesion

Dataset counts, class names, and split details should be checked against the actual data and code before publication.
Limitations 

SkinStackNet is an educational and experimental machine learning project. It is not a clinically validated medical device and must not be used as a substitute for professional medical evaluation.

Model performance on a particular dataset does not guarantee reliable predictions on unseen images or different patient populations.

📌 Publication Status

This repository documents an academic team project. The work has not been published as a peer-reviewed research paper.

Please refer to the original dataset, pretrained model, and third-party software licenses before redistributing any associated materials.

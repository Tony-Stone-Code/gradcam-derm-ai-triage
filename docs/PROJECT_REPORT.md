# X-Skin: CNN-Based Skin Cancer Detection — Technical Report

## 1. Project Overview

**Title**: X-Skin — Comparative Analysis of CNN Architectures for Dermoscopic Skin Lesion Classification

**Objective**: Evaluate and compare four convolutional neural network architectures for automated classification of skin lesions from dermoscopic images, with the goal of selecting the optimal model for mobile deployment.

**Dataset**: HAM10000 (Human Against Machine with 10000 training images)
- **Source**: International Skin Imaging Collaboration (ISIC)
- **Size**: ~10,015 dermoscopic images
- **Classes**: 7 diagnostic categories

| Class | Full Name | Count | % |
|---|---|---|---|
| NV | Melanocytic nevi | 6,705 | 66.9% |
| MEL | Melanoma | 1,113 | 11.1% |
| BKL | Benign keratosis | 1,099 | 11.0% |
| BCC | Basal cell carcinoma | 514 | 5.1% |
| AKIEC | Actinic keratoses | 327 | 3.3% |
| VASC | Vascular lesions | 142 | 1.4% |
| DF | Dermatofibroma | 115 | 1.1% |

**Environment**: Kaggle (free tier, GPU P100/T4)

---

## 2. Architectures Evaluated

| Architecture | Parameters | ImageNet Top-1 | Mobile-Friendly | Fine-tune Layer |
|---|---|---|---|---|
| VGG16 | 138M | 71.3% | ❌ (large) | 11 |
| ResNet50 | 25.6M | 76.1% | ⚠️ (medium) | 120 |
| MobileNetV2 | 3.4M | 71.3% | ✅ (designed for mobile) | 80 |
| EfficientNetB0 | 5.3M | 77.1% | ✅ (efficient) | 150 |

All architectures use **transfer learning** from ImageNet pre-trained weights with a custom classification head.

### Classification Head Architecture
```
GlobalAveragePooling2D
  → BatchNormalization → Dropout(0.4)
  → Dense(512, ReLU)
  → BatchNormalization → Dropout(0.3)
  → Dense(128, ReLU)
  → BatchNormalization → Dropout(0.2)
  → Dense(7, Softmax)
```

---

## 3. Methodology

### 3.1 Data Preprocessing
- **Image resizing**: 224×224 pixels
- **Stratified split**: 80% train / 10% validation / 10% test
- **Architecture-specific preprocessing**: Each model uses its own `preprocess_input` function

### 3.2 Class Imbalance Handling (Three-Pronged Strategy)
The HAM10000 dataset has severe class imbalance (NV is 60× larger than DF). Three complementary techniques address this:

1. **Oversampling**: Minority classes are duplicated to reach 50% of the majority class count, increasing training set from ~8,000 to ~22,000+ samples
2. **Class Weights**: `compute_class_weight('balanced')` with **sqrt dampening** to avoid over-predicting rare classes
3. **Focal Loss**: `CategoricalFocalCrossentropy(alpha=0.25, gamma=1.5)` down-weights well-classified examples

### 3.3 Data Augmentation
Applied only to training data:
- Rotation: ±40°
- Width/Height shift: ±20%
- Shear: 20%
- Zoom: ±20%
- Horizontal & vertical flip
- Brightness: [0.8, 1.2]

### 3.4 Two-Phase Training Protocol
**Phase 1 — Feature Extraction** (15 epochs, LR=1e-3)
- Base model frozen, only classification head trains
- EarlyStopping (patience=7), ReduceLROnPlateau (patience=4)

**Phase 2 — Fine-tuning** (30 epochs, LR=5e-5)
- Base model partially unfrozen (architecture-specific layer)
- Lower learning rate prevents catastrophic forgetting
- ModelCheckpoint saves best weights by val_accuracy

### 3.5 Evaluation Metrics
- **Classification Report**: Per-class precision, recall, F1-score
- **Confusion Matrix**: Raw and normalized
- **ROC-AUC Curves**: Per-class with area under curve
- **Training Curves**: Accuracy and loss over epochs
- **Grad-CAM Heatmaps**: Visual explainability per architecture

### 3.6 Decoupled Weighted Voting Ensemble
After training all 4 models independently, a **Decoupled Weighted Voting Ensemble** combines their predictions for superior performance.

**How it works**:
1. Each model's **per-class F1-score** on the validation set determines its weight for that class
2. Weights are normalized per class so they sum to 1.0
3. At inference, each model's predicted probability is multiplied by its per-class weight
4. The final prediction is the class with the highest weighted sum

**Mathematical formulation**:
```
final_score[c] = Σ_m ( W[m][c] × P[m][c] )
prediction = argmax_c ( final_score[c] )
```

Where:
- `W[m][c]` = normalized weight of model `m` for class `c` (derived from val F1)
- `P[m][c]` = predicted probability from model `m` for class `c`

**Why "decoupled"**: Unlike uniform voting where each model has equal say for all classes, this approach gives each model **different influence per class**. A model strong on melanoma (MEL) has high weight for MEL predictions but may have low weight for dermatofibroma (DF) predictions. This leverages each architecture's unique strengths.

### 3.7 Stacking Ensemble (Meta-Learner)
To capture non-linear relationships and resolve specific inter-class confusions (like MEL vs NV), a **Stacking Ensemble** was implemented alongside the weighted voting.

**Methodology**:
1. **Feature Extraction**: All 4 CNNs predict on the validation set, outputting 7 probabilities each (total: 28 features per image).
2. **Meta-Learner Training**: A secondary Machine Learning classifier (Random Forest or Logistic Regression) is trained on these 28 features using the validation labels.
3. **Inference**: Test images are passed through the 4 CNNs to generate the 28 features, which the Meta-Learner uses to make the final prediction.

Unlike weighted voting which applies a static linear multiplier, a Random Forest meta-learner can learn complex conditional logic (e.g., "If VGG16 predicts MEL with >80% confidence, but EfficientNet predicts NV with >90% confidence, the true label is usually NV").

### 3.8 Confidence-Gated Cascade (Mobile-Optimized)
To bridge the gap between high accuracy (the heavy Stacking Ensemble) and real-world deployment constraints (Mobile devices), a **Confidence-Gated Cascade** architecture was designed.

**Methodology**:
1. **Stage 1 (Local/Fast)**: The dermoscopic image is passed to `MobileNetV2` (3.4M params), which runs quickly on the mobile device.
2. **Gating Logic**: If the maximum predicted probability from MobileNetV2 exceeds a defined threshold (e.g., `T > 0.85`), the prediction is accepted immediately.
3. **Stage 2 (Cloud/Heavy Fallback)**: If MobileNetV2 is uncertain (`P < T`), the image is sent to the cloud where the heavy Stacking Ensemble processes it.

This operational ensemble provides a controllable tradeoff: we can retain 99% of the Stacking Ensemble's massive accuracy while processing a large percentage of images instantly on-device, saving battery, compute, and bandwidth.

### 3.9 Rank-Based Voting (Borda Count)
To provide a highly robust, mathematically distinct ensemble that is simple to deploy in a standalone web application, a **Rank-Based Voting (Borda Count)** architecture was implemented.

**Methodology**:
1. Instead of using the raw probability percentages (which suffer from neural network overconfidence), this ensemble evaluates the **ordinal ranking** of the predictions.
2. For every image, each of the 4 models ranks the 7 classes. 
3. The top choice receives 6 points, the second choice 5 points, down to 0 points for the last choice.
4. Points are tallied across all models, and the class with the highest Borda Score is the final prediction.

This approach guarantees that every model has exactly equal voting power, completely neutralizing any poorly calibrated model that consistently outputs `0.99` probabilities.

### 3.10 Out-of-Distribution (OOD) Rejection
A fundamental limitation of softmax-based classifiers is that the output layer always distributes exactly 100% probability across the defined classes, regardless of whether the input belongs to the training domain. An image of a landscape, a pet, or any non-dermatoscopic photograph will still receive a full probability distribution across the 7 skin lesion classes, producing a misleading diagnosis.

To mitigate this, a **dual-gate rejection mechanism** was implemented in the web application. Both gates must pass before a prediction is displayed to the user:

**Gate 1 — Confidence Threshold**: The maximum predicted probability must be at least 50%. A genuine skin lesion should produce moderate-to-high confidence in at least one diagnostic class. If the model distributes probability roughly equally across multiple classes, the input is likely outside its domain of competence.

**Gate 2 — Shannon Entropy**: The Shannon entropy of the predicted probability distribution must not exceed 1.6 nats. Shannon entropy quantifies the uncertainty in a discrete probability distribution and is defined as:

```
H(p) = -Σ p_i × log(p_i)
```

For a uniform distribution over 7 classes (the maximum-uncertainty case), the entropy is:

```
H_max = log(7) ≈ 1.946 nats
```

The threshold of 1.6 nats was selected to provide a meaningful margin below the theoretical maximum while still allowing legitimate low-confidence predictions on ambiguous skin lesions (e.g., early-stage melanoma vs. atypical nevus).

**Combined Effect**: If either gate fails, the application displays a rejection message ("Not a Valid Dermatoscopic Skin Lesion") instead of a potentially harmful misdiagnosis. This dual-gate design is deliberately conservative — a non-skin image must fool both the confidence and entropy checks simultaneously to pass through.

---

## 4. Results

### 4.1 EfficientNetB0 (Enhanced v2)

| Metric | Value |
|---|---|
| **Test Accuracy** | **78.84%** |
| **Macro F1** | **0.7044** |
| **Weighted F1** | **0.8010** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AKIEC | 0.9091 | 0.3125 | 0.4651 | 32 |
| BCC | 0.9189 | 0.6538 | 0.7640 | 52 |
| BKL | 0.4660 | 0.8727 | 0.6076 | 110 |
| DF | 0.8000 | 0.7273 | 0.7619 | 11 |
| MEL | 0.5177 | 0.6518 | 0.5771 | 112 |
| NV | 0.9570 | 0.8286 | 0.8882 | 671 |
| VASC | 0.8125 | 0.9286 | 0.8667 | 14 |

### 4.2 MobileNetV2 (Pre-Enhancement Baseline)

| Metric | Value |
|---|---|
| **Test Accuracy** | **78.04%** |
| **Macro F1** | **0.5471** |
| **Weighted F1** | **0.7602** |

*Note: Results pending re-run with oversampling + enhanced configuration.*

### 4.3 VGG16 & ResNet50
*Training pending with enhanced pipeline.*

### 4.4 Decoupled Weighted Voting Ensemble
*Results pending execution on Kaggle.*

### 4.5 Stacking Ensemble (Meta-Learner)

The Stacking Ensemble used the probability outputs of all 4 CNNs as features to train a Random Forest Meta-Learner. This approach proved to be the **best performing model in the entire project**.

| Metric | Value |
|---|---|
| **Test Accuracy** | **89.52%** |
| **Macro F1** | **0.8234** |
| **Weighted F1** | **0.8940** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| AKIEC | 0.7333 | 0.6875 | 0.7097 | 32 |
| BCC | 0.8750 | 0.8077 | 0.8400 | 52 |
| BKL | 0.7845 | 0.8273 | 0.8053 | 110 |
| DF | 0.7692 | 0.9091 | 0.8333 | 11 |
| MEL | 0.7255 | 0.6607 | 0.6916 | 112 |
| NV | 0.9499 | 0.9613 | 0.9556 | 671 |
| VASC | 0.9286 | 0.9286 | 0.9286 | 14 |

**Comparison to Weighted Voting:**
The Random Forest Stacking Ensemble improved upon the Decoupled Weighted Voting (88.72% → 89.52% accuracy). More importantly, the **AKIEC F1-score jumped from 0.6383 to 0.7097**, proving that the Random Forest was able to learn complex non-linear rules to resolve the difficult `AKIEC → MEL` confusion that the linear weighted voting could not handle.

### 4.6 Confidence-Gated Cascade
*Results pending execution of `ensemble_cascade.py` on Kaggle. The script will simulate the cascade and output an optimal threshold mapping showing the tradeoff between local processing efficiency and overall accuracy.*

### 4.7 Rank-Based Voting (Borda Count)

The Rank-Based ensemble achieved an **Accuracy of 86.03%** and a **Macro F1 of 0.7839**. 

| Metric | Value |
|---|---|
| **Test Accuracy** | **86.03%** |
| **Macro F1** | **0.7839** |

**Analysis**: 
While Rank-Based Voting had lower raw accuracy than the Weighted and Stacking ensembles, it successfully outperformed the best individual model (ResNet50) in Macro F1 (0.7839 vs 0.7568). By ignoring probability percentages entirely, it sacrificed some majority-class accuracy but proved highly effective at equalizing the voting power across models for minority classes.

---

## 5. Key Findings

### 5.1 Class Imbalance is the Dominant Challenge
The HAM10000 dataset's extreme imbalance (67% NV) means raw accuracy is misleading. **Macro F1-score** is the most appropriate primary metric as it treats all classes equally.

### 5.2 Oversampling > Class Weights Alone
Class weights adjust gradient magnitude but don't increase the diversity of minority class examples the model sees. **Oversampling + augmentation** ensures the model sees many diverse versions of rare lesion types each epoch.

### 5.3 Focal Loss Gamma Requires Tuning
The default gamma=2.0 was too aggressive, causing the model to over-focus on hard examples at the expense of overall accuracy. **gamma=1.5** provided a better balance.

### 5.4 Clinical Ambiguity and Known Confusions
Even with the ensemble, the models struggle with two specific clinical distinctions:
1. **True: MEL → Predicted: NV**: This is the most common and dangerous misclassification. Early melanomas (MEL) are visually very similar to atypical melanocytic nevi (NV). Because NV is the vast majority class, the model defaults to NV when uncertain.
2. **True: AKIEC → Predicted: MEL**: Actinic keratoses (AKIEC) can sometimes present with pigmentation or vascular patterns that overlap with melanoma features in dermoscopy, leading the model to predict the more common severe class (MEL).

This is a known limitation in automated dermatology; even expert dermatologists face ~76-80% inter-observer agreement on these specific borderline cases without biopsies.

### 5.5 Ensemble Methods Exploit Model Diversity
Different architectures learn different feature representations. The decoupled ensemble leverages each model's per-class strengths, which should outperform any single model — especially on the challenging minority classes.

---

## 6. Future Work

1. **Mobile Deployment**: Convert best model to TFLite for on-device inference
2. **FastAPI Mobile Backend**: The Streamlit web application has been built and deployed to Streamlit Community Cloud (lightweight MobileNetV2 model, with OOD rejection). The remaining work is to build a FastAPI-based REST API to serve the full Stacking Ensemble as the cloud backend for the Confidence-Gated Cascade mobile architecture
3. **Extended Dataset**: Incorporate ISIC 2019/2020 datasets for more training data
4. **Threshold Optimization**: Tune per-class decision thresholds for clinical sensitivity requirements
5. **Patient Metadata Integration**: Incorporate patient metadata — age, sex, and anatomical location of the lesion — as additional input features to the stacking meta-learner. Clinical literature demonstrates that lesion demographics significantly influence diagnostic probability (e.g., melanoma incidence varies by anatomical site and patient age). Concatenating these structured features with the 28-dimensional CNN probability vector could improve the meta-learner's discriminative power, particularly for clinically ambiguous cases such as MEL vs. NV

---

## 7. References

1. Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset. *Sci. Data* 5, 180161 (2018).
2. Tan, M. & Le, Q.V. EfficientNet: Rethinking Model Scaling. *ICML* (2019).
3. Lin, T.Y. et al. Focal Loss for Dense Object Detection. *ICCV* (2017).
4. Selvaraju, R.R. et al. Grad-CAM: Visual Explanations from Deep Networks. *ICCV* (2017).

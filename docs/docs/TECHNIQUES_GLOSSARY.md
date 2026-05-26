# Techniques Explained (Glossary)

This document explains the advanced Machine Learning and Deep Learning techniques used in the X-Skin project in plain English. This is useful for explaining the project to non-technical stakeholders or defense committees.

---

## Data Imbalance Solutions

The HAM10000 dataset is highly imbalanced (e.g., 6700 Nevus images, but only 115 Dermatofibroma images). If we train a model on this directly, the model will just guess "Nevus" every time and achieve 67% accuracy by doing nothing. We used two techniques to fix this:

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
* **The Problem**: We don't have enough images of rare cancers.
* **The Solution**: SMOTE looks at the existing images of rare cancers in the "feature space" (how the computer sees them) and mathematically creates brand new, synthetic examples by interpolating between them. 
* **The Result**: The model has equal amounts of data to learn from for every class.

### 2. Class Weights (Cost-Sensitive Learning)
* **The Problem**: Even with SMOTE, the original data distribution affects how the model updates its internal math (gradients).
* **The Solution**: We tell the loss function to "penalize" the model much harder if it gets a rare cancer wrong, and penalize it less if it gets a common Nevus wrong.
* **The Result**: The model pays closer attention to the rare classes.

---

## Ensembling Techniques

Ensembling is the process of combining multiple different AI models to make a single, better decision—similar to a medical board of different specialists consulting on a patient.

### 1. Decoupled Weighted Soft-Voting
* **How it works**: Every model gives a percentage guess (e.g., "I am 90% sure this is Melanoma"). We multiply that percentage by a specific "weight" based on how good that model is at detecting Melanoma.
* **Why "Decoupled"?**: A model might be an expert at Melanoma but terrible at Basal Cell Carcinoma. Decoupled voting gives the model a high multiplier for Melanoma, but a low multiplier for BCC.

### 2. Stacking (Meta-Learning)
* **How it works**: We take the output percentages from the 4 models and feed them into a brand new, secondary AI (a Random Forest). 
* **The Advantage**: The Random Forest can learn complex, non-linear rules. It can learn things like: *"If VGG16 is highly confident it's Melanoma, but EfficientNet thinks it's a Nevus, EfficientNet is usually right in this specific scenario, so choose Nevus."*

### 3. Confidence-Gated Cascade
* **How it works**: A multi-stage pipeline designed to save battery and computing power on mobile phones. The image goes to the fastest model first (MobileNet). If MobileNet is highly confident, we accept the answer immediately. If it is uncertain, we send the image to a cloud server to run the heavy, accurate Stacking ensemble.

### 4. Borda Count (Rank-Based Voting)
* **How it works**: Ignores the raw probability percentages entirely. Instead, each model simply ranks the 7 diseases. 1st place gets 6 points, 2nd gets 5 points, etc. The class with the most points wins.
* **The Advantage**: Prevents a single overconfident (but incorrect) model from hijacking the final decision.

---

## Explainability (XAI)

### Grad-CAM (Gradient-weighted Class Activation Mapping)
* **The Problem**: Deep Learning models are "Black Boxes." A doctor cannot trust an AI if the AI cannot explain *why* it made a diagnosis.
* **The Solution**: Grad-CAM tracks the math backwards from the final prediction to the last visual layer of the network. It creates a "heatmap" showing exactly which pixels the AI was looking at when it made its decision.
* **The Result**: If the AI diagnoses Melanoma, and the heatmap highlights the dark, asymmetrical border of the mole, the doctor knows the AI is using medically sound reasoning. If the heatmap highlights a skin hair or a ruler mark next to the mole, the doctor knows the AI is biased and should be ignored.

---

## Out-of-Distribution (OOD) Rejection

### The Problem: Softmax Forces a Diagnosis

The softmax function used in the final layer of a classification neural network always produces probabilities that sum to exactly 1.0. This is a fundamental mathematical constraint, not an indication of genuine confidence. If a user uploads a photograph of a cat, a car, or a blank white image, the model will still distribute 100% of its probability across the seven skin lesion classes and return a "diagnosis." Without additional safeguards, the system has no ability to say "I do not know what this is."

### The Solution: A Three-Gate Validation System

Before displaying any result, the system evaluates the model's output probabilities against three independent checks. All three must pass for a prediction to be accepted.

1. **NV Bias Correction (85% threshold for NV)**: If the model's top prediction is NV (Melanocytic Nevus), the system demands at least 85% confidence before accepting it. This gate exists because the HAM10000 training set is approximately 67% NV, which creates a massive class prior bias in the learned softmax distribution. In practice, this means the model "sees" skin-like texture in almost any input — including photographs of human faces, bare arms, or random skin — and defaults to NV with moderate confidence (typically 60--70%). The 85% threshold forces the model to be extremely certain before an NV classification is permitted, preventing the dominant training class from acting as a catch-all for non-lesion images.

2. **Confidence Threshold (70%)**: The system examines the highest predicted probability. If no single class receives at least 70% of the total confidence, the model has not identified a dominant candidate. The threshold was raised from the original 50% to 70% because a genuine dermatoscopic lesion should produce strong activation in a well-trained classifier.

3. **Shannon Entropy (threshold: 1.2)**: Shannon Entropy measures how "spread out" or disordered a probability distribution is. In plain English, entropy answers the question: *"Is the model pointing at one answer, or is it shrugging across all seven?"*
   * **Formula**: H = -sum(p * log(p)) for each class probability p
   * A perfectly confident prediction (100% on one class) has an entropy of **0.0**.
   * A completely uniform guess across 7 classes has the maximum entropy of **log(7) = 1.946**.
   * The threshold was tightened from 1.6 to **1.2** to enforce a stricter spread requirement. Distributions exceeding 1.2 entropy indicate that the model's probability mass is too dispersed to represent a genuine clinical prediction.

### Why the NV Bias Correction Gate Is Necessary

The NV Bias Correction gate was introduced after a specific failure case was identified during testing: the model predicted NV (Melanocytic Nevus) on a photograph of a human face. The face contained skin-like colour and texture, which was sufficient to activate the NV-dominant features learned during training. Because 67% of the training data consists of NV images, the softmax output layer inherits a strong statistical prior toward NV. Any input that contains skin — even healthy, non-lesion skin — will trigger moderate NV confidence (typically 60--70%), which was enough to pass the original 50% confidence gate.

The 85% NV-specific threshold functions as a form of post-hoc calibration that compensates for dataset imbalance at inference time. Rather than retraining the model or rebalancing the dataset (which would affect all other class boundaries), this targeted gate addresses the single most problematic failure mode without disrupting the classifier's performance on the remaining six classes.

If any check fails, the system displays a red error banner: *"Image Rejected: Not a Valid Dermatoscopic Skin Lesion"* and provides guidance to the user.

---

## Lazy Loading (Deferred Model Initialization)

### The Problem: Server Timeouts on Cloud Platforms

The X-Skin ensemble loads four large CNN architectures (VGG16, ResNet50, MobileNetV2, and EfficientNetB0), each with millions of parameters. When all four models are loaded into memory at application startup (boot time), the initialisation process can take 30 seconds or longer. Cloud hosting platforms such as HuggingFace Spaces enforce strict startup timeouts, and exceeding these limits causes the application to be terminated before it ever becomes available to users.

### The Solution: Load on Demand

Rather than loading all models when the server boots, the system defers model loading until the user actually requests an inference by clicking "Analyze Image." The first analysis request incurs a brief loading delay, but subsequent requests use the already-loaded models from memory. This approach eliminates boot-time timeouts entirely, ensuring the web application starts quickly and remains responsive on resource-constrained cloud environments.

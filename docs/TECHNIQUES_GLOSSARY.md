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

### The Solution: A Two-Pronged Validation Gate

Before displaying any result, the system evaluates the model's output probabilities against two independent checks. Both must pass for a prediction to be accepted.

1. **Confidence Threshold (50%)**: The system examines the highest predicted probability. If no single class receives at least 50% of the total confidence, the model has not identified a dominant candidate. The prediction is rejected because a clinically meaningful diagnosis should concentrate the majority of probability mass on a single class.

2. **Shannon Entropy (threshold: 1.6)**: Shannon Entropy measures how "spread out" or disordered a probability distribution is. In plain English, entropy answers the question: *"Is the model pointing at one answer, or is it shrugging across all seven?"*
   * **Formula**: H = -sum(p * log(p)) for each class probability p
   * A perfectly confident prediction (100% on one class) has an entropy of **0.0**.
   * A completely uniform guess across 7 classes has the maximum entropy of **log(7) = 1.946**.
   * The threshold of **1.6** was chosen to provide a reasonable margin below the theoretical maximum. Any distribution exceeding 1.6 entropy indicates the model is nearly uniformly confused, and the input is likely not a valid skin lesion.

If either check fails, the system displays a red error banner: *"Image Rejected: Not a Valid Dermatoscopic Skin Lesion"* and provides guidance to the user.

---

## Lazy Loading (Deferred Model Initialization)

### The Problem: Server Timeouts on Cloud Platforms

The X-Skin ensemble loads four large CNN architectures (VGG16, ResNet50, MobileNetV2, and EfficientNetB0), each with millions of parameters. When all four models are loaded into memory at application startup (boot time), the initialisation process can take 30 seconds or longer. Cloud hosting platforms such as HuggingFace Spaces enforce strict startup timeouts, and exceeding these limits causes the application to be terminated before it ever becomes available to users.

### The Solution: Load on Demand

Rather than loading all models when the server boots, the system defers model loading until the user actually requests an inference by clicking "Analyze Image." The first analysis request incurs a brief loading delay, but subsequent requests use the already-loaded models from memory. This approach eliminates boot-time timeouts entirely, ensuring the web application starts quickly and remains responsive on resource-constrained cloud environments.

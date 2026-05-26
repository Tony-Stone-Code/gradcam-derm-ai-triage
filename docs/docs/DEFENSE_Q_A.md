# Final Year Project Defense Q&A

This document contains a curated list of difficult questions a defense committee might ask about the X-Skin project, along with professional, well-reasoned answers.

---

### Q1: Why did you choose these specific 4 CNN architectures?
**Answer**: We wanted architectural diversity to maximize the effectiveness of our ensemble. 
* **VGG16** provides a traditional, deep sequential baseline.
* **ResNet50** utilizes skip connections (residual blocks) to capture very deep spatial hierarchies without vanishing gradients.
* **MobileNetV2** uses depthwise separable convolutions, focusing on creating a lightweight, fast model for our mobile deployment goal.
* **EfficientNetB0** uses compound scaling to provide the absolute highest accuracy per parameter.
By combining models with fundamentally different ways of "seeing" the image, the ensemble can cover the blind spots of any single model.

### Q2: The HAM10000 dataset is notoriously imbalanced (mostly Nevi). How did you handle this?
**Answer**: We tackled class imbalance on two fronts:
1. **At the data level**: We applied SMOTE (Synthetic Minority Over-sampling Technique) to algorithmically generate synthetic feature examples of the rare classes, ensuring the model had a balanced distribution to learn from.
2. **At the algorithmic level**: We applied computed Class Weights to the Cross-Entropy loss function. If the model incorrectly guessed a rare class (like Dermatofibroma), the loss penalty was multiplied, forcing the optimizer to pay close attention to the minority classes rather than just guessing "Nevus" every time.

### Q3: Your models struggled with the "AKIEC vs. MEL" and "MEL vs. NV" distinctions. Why does this happen, and how did you fix it?
**Answer**: This confusion is actually clinically accurate. Early melanomas (MEL) often share macroscopic features with atypical nevi (NV), and actinic keratoses (AKIEC) can present as pigmented lesions mimicking melanoma. 
To resolve this mathematically, we built a **Random Forest Stacking Ensemble**. Unlike standard soft-voting, the Random Forest acts as a Meta-Learner. It learned specific non-linear conditions—for example, if VGG16 is highly confident about MEL, but EfficientNet is certain it's NV, the Random Forest learns which model's "opinion" is historically more accurate for that specific feature combination. This specific technique boosted our AKIEC F1-score from 0.63 up to 0.70.

### Q4: Why implement 4 different ensemble techniques instead of just taking the average?
**Answer**: Standard averaging (Uniform Soft Voting) is brittle. If one model is poorly calibrated and outputs 99% confidence for the wrong answer, it ruins the average. We wanted to engineer robust solutions:
* **Decoupled Weighted Voting**: Applies linear multipliers based on validation F1-scores, rewarding models only for the specific diseases they are good at detecting.
* **Stacking (Meta-Learner)**: Learns complex non-linear corrections between the base models.
* **Rank-Based Voting (Borda Count)**: Completely ignores probability percentages to neutralize overconfident models, looking only at the ordinal ranking of the predictions.
* **Confidence-Gated Cascade**: An operational ensemble designed to simulate the battery/compute constraints of our ultimate Mobile App deployment goal.

### Q5: How exactly does the Confidence-Gated Cascade work?
**Answer**: Running all 4 models on a mobile phone simultaneously would drain the battery and cause massive latency. The Cascade solves this by acting as a filter. The image is processed instantly on the phone by our lightweight `MobileNetV2` model. If MobileNet's confidence exceeds a high threshold (e.g., 90%), we accept the result immediately. If it is uncertain, we escalate the image to a cloud server where the heavy Stacking Ensemble processes it. This allowed us to maintain 99% of our maximum accuracy while processing over 60% of images locally on the phone.

### Q6: Deep Learning in healthcare faces scrutiny for being a "black box". How can a doctor trust your model?
**Answer**: We integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) directly into our inference pipeline. Grad-CAM traces the gradients of the predicted class back to the final convolutional layer, producing a spatial heatmap of what the model was "looking at". If the model diagnoses Melanoma, the doctor can look at the heatmap. If the heatmap highlights the asymmetrical border of the lesion, the doctor knows the AI's reasoning aligns with dermatological science. If it highlights a ruler marking in the corner of the image, the doctor knows to disregard the prediction.

### Q7: If you had 6 more months to work on this, what would you add?
**Answer**: 
1. **Patient Metadata**: We currently only use the image. Feeding patient age, sex, and anatomical location into the dense layers of the network (or directly into the Meta-Learner) would significantly boost accuracy, as diseases like BCC are highly correlated with age and sun-exposed areas.
2. **Test-Time Augmentation (TTA)**: Passing the same image through the model 5 times at different rotations and averaging the result to increase robustness against poor camera angles.
3. **ISIC 2019/2020 Extended Datasets**: Incorporating the larger ISIC 2019 and ISIC 2020 challenge datasets, which introduce additional diagnostic categories and a significantly greater volume of labelled dermatoscopic images, to further improve generalisation and reduce class-level confusion.

### Q8: Your model predicts a diagnosis even when the input is not a skin image. How did you handle this?
**Answer**: This is a fundamental limitation of the softmax activation function. Because softmax always produces probabilities that sum to 1.0, the model is mathematically forced to distribute 100% of its confidence across the seven known skin lesion classes, regardless of whether the input is actually a skin lesion. A photograph of a car, a pet, or a blank image will still receive a seemingly confident diagnosis.

To address this, we implemented a **three-gate Out-of-Distribution (OOD) Rejection mechanism** in the inference pipeline. Before any prediction is displayed to the user, the system evaluates three independent checks:

1. **NV Bias Correction (85% for NV)**: If the model's top prediction is NV (Melanocytic Nevus), the system requires at least 85% confidence before accepting it. The HAM10000 dataset is approximately 67% NV, which creates a strong prior bias causing the model to default to NV on any skin-like texture — including human faces, arms, or other non-lesion photographs. The elevated 85% threshold prevents this dominant class from acting as a catch-all.
2. **Confidence Threshold (70%)**: The system examines the highest predicted probability across all seven classes. If no single class exceeds 70%, the model has failed to identify a dominant candidate, and the input is rejected. The threshold was raised from 50% to 70% because genuine dermatoscopic lesions should produce strong classifier activation.
3. **Shannon Entropy Check (threshold: 1.2)**: The system computes the Shannon Entropy of the probability distribution using H = -sum(p * log(p)). A uniform distribution over 7 classes yields the theoretical maximum entropy of log(7) = 1.946. If the computed entropy exceeds 1.2, the model's uncertainty is too high, and the input is rejected. The threshold was tightened from 1.6 to 1.2 to enforce a stricter spread requirement.

All three conditions must be satisfied for a prediction to be accepted. When rejection triggers, the user sees a red error banner reading "Image Rejected: Not a Valid Dermatoscopic Skin Lesion" along with guidance to upload a properly captured dermatoscopic image. No diagnosis, confidence score, or Grad-CAM heatmap is shown.

### Q9: Why did you choose Shannon Entropy over other uncertainty measures?
**Answer**: Shannon Entropy is a well-established information-theoretic measure with several properties that make it particularly suitable for this application.

First, it captures the entire "spread" of a probability distribution in a single scalar value, making it computationally trivial to evaluate at inference time. A confident prediction, where one class dominates, produces low entropy (approaching 0.0). A confused prediction, where probability mass is spread roughly equally across all classes, produces high entropy (approaching the theoretical maximum of log(7) = 1.946 for our seven-class problem).

Second, unlike simpler alternatives such as using only the maximum predicted probability, entropy considers the shape of the entire distribution. Two predictions might both have a maximum confidence of 55%, but one could concentrate the remaining 45% on a single alternative class (low entropy, relatively confident), while the other spreads it evenly across all six remaining classes (high entropy, genuinely confused). Entropy distinguishes between these two cases in a way that a single confidence threshold alone cannot.

The threshold of 1.2 was selected to provide a strict margin below the theoretical maximum of 1.946. The original threshold of 1.6 was found to be too permissive, allowing some ambiguous non-dermatoscopic inputs to pass through. Distributions exceeding 1.2 are sufficiently dispersed that accepting them as valid clinical predictions would be irresponsible.

### Q10: Why does the NV class need a separate, higher confidence threshold?
**Answer**: The NV (Melanocytic Nevus) class requires a dedicated 85% confidence threshold — rather than the standard 70% applied to all other classes — because of the extreme class imbalance in the HAM10000 training dataset.

The training set is composed of approximately 67% NV images, which means roughly two out of every three images the model learned from were Melanocytic Nevi. This creates a massive class prior bias in the learned softmax distribution: the model's internal feature representations are disproportionately tuned to NV-like patterns, and its output layer carries a strong statistical tendency to assign probability mass to NV.

The practical consequence is that the softmax output inherits this bias, and any input containing skin-like texture — even if it is not a skin lesion — will trigger moderate NV confidence. During testing, a photograph of a human face was submitted to the model, and it returned a prediction of NV with approximately 65% confidence. The face contained natural skin colour and texture, which was sufficient to activate the NV-dominant features. Similarly, photographs of bare arms, shoulders, or any exposed skin surface would produce the same behaviour, with NV confidence typically falling in the 60--70% range.

Under the original 50% confidence gate, all of these non-lesion inputs would have been accepted as valid NV predictions, which is clinically dangerous and erodes user trust. The 85% NV-specific threshold forces the model to demonstrate very high certainty before an NV classification is permitted. This functions as a form of post-hoc calibration that compensates for dataset imbalance at inference time. Rather than retraining the entire model or artificially rebalancing the dataset — either of which would disturb all other class decision boundaries — this targeted gate addresses the single most problematic failure mode (NV defaulting) without affecting the classifier's performance on the remaining six diagnostic classes.

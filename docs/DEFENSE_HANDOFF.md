# X-Skin Defense Handoff Guide

Quick reference for capstone defense preparation.

---

## Project at a Glance

| Item | Details |
|---|---|
| **Project** | X-Skin — CNN-Based Skin Cancer Detection |
| **Dataset** | HAM10000 (10,015 dermoscopic images, 7 classes) |
| **Architectures** | VGG16, ResNet50, MobileNetV2, EfficientNetB0 |
| **Best Model** | EfficientNetB0 (78.8% accuracy, 0.70 macro F1) |
| **Deployment Target** | Mobile devices via TFLite |
| **Inference API** | Streamlit / FastAPI (planned) |

---

## File Structure

```
Skin-Cancer-Detection-main/
├── CNN_skincancer_FIX.ipynb    # Original notebook (legacy)
├── training/
│   ├── train_vgg16.py          # VGG16 Kaggle training script
│   ├── train_resnet50.py       # ResNet50 Kaggle training script
│   ├── train_mobilenetv2.py    # MobileNetV2 Kaggle training script
│   └── train_efficientnetb0.py # EfficientNetB0 Kaggle training script
├── docs/
│   ├── PROJECT_REPORT.md       # Full technical report
│   ├── ENHANCEMENT_LOG.md      # Log of all optimizations
│   └── DEFENSE_HANDOFF.md      # This file
└── README.md                   # Project overview
```

---

## How to Run (Kaggle)

1. Create a new Kaggle notebook
2. Add dataset: **"skin-cancer-mnist-ham10000"**
3. Settings → **GPU** accelerator
4. Copy code from the `.py` file (each `# %%` = one notebook cell)
5. Run all cells (~1.5-2.5 hours per model)

### Output per Model
```
results_{model_name}/
├── {model}_best.keras      # Best checkpoint
├── {model}_final.keras     # Final model
├── {model}_cm.png          # Confusion matrices
├── {model}_roc.png         # ROC-AUC curves
├── {model}_curves.png      # Training curves
└── {model}_gradcam.png     # Grad-CAM heatmaps
```

---

## Key Techniques to Discuss in Defense

### 1. Transfer Learning
- Pre-trained ImageNet weights provide strong feature extractors
- Two-phase training: frozen head → fine-tuned base
- Prevents overfitting on small medical dataset

### 2. Class Imbalance Strategy (Most Important!)
Three complementary techniques:

| Technique | What It Does | Why Needed |
|---|---|---|
| **Oversampling** | Duplicates minority rows to 50% of majority count | Increases diversity of rare class examples |
| **Sqrt-dampened class weights** | Penalizes majority class misclassification less aggressively | Prevents over-predicting rare classes |
| **Focal loss (γ=1.5)** | Down-weights easy examples, focuses on hard ones | Handles sample-level difficulty |

**Key insight**: Class weights alone are NOT sufficient. The model needs to actually *see* more diverse examples of rare classes (via oversampling + augmentation).

### 3. Grad-CAM Explainability
- Visual proof that the model looks at the lesion, not background artifacts
- One heatmap per class demonstrates model attention regions
- Critical for clinical trust and academic rigor

### 4. Architecture Comparison
Be prepared to explain why each architecture was chosen:

| Architecture | Strengths | Weaknesses |
|---|---|---|
| **VGG16** | Simple, well-understood | Very large (138M params), slow |
| **ResNet50** | Skip connections prevent vanishing gradients | Moderate size (25.6M) |
| **MobileNetV2** | Lightweight, designed for mobile | Lower capacity, depthwise separable convs |
| **EfficientNetB0** | Compound scaling, best accuracy/params ratio | Slightly complex architecture |

---

## Anticipated Defense Questions & Answers

### Q: "Why not use a larger dataset?"
**A**: HAM10000 is the gold-standard benchmark for dermoscopic classification (Tschandl et al., 2018). It's peer-reviewed, clinically validated, and widely used in literature. Future work can incorporate ISIC 2019/2020 datasets.

### Q: "Why is accuracy only ~79% and not 95%+?"
**A**: Three reasons:
1. **Class imbalance**: 67% of data is one class (NV). Raw accuracy is misleading — macro F1 (0.70) is the fairer metric
2. **Inherent difficulty**: Even dermatologists disagree on borderline cases (inter-observer agreement ~76-80%)
3. **Limited data**: Only ~90-260 training images for rare classes, even after oversampling

### Q: "Why does the model confuse AKIEC with MEL, or MEL with NV?"
**A**: This reflects real-world clinical ambiguity:
- **MEL vs NV**: Early melanomas (MEL) look almost identical to atypical nevi (NV). Since NV is the vast majority class, the model falls back to it when uncertain.
- **AKIEC vs MEL**: Actinic keratoses can have pigmentation or vascular patterns that mimic melanoma features in dermoscopy. Even trained dermatologists require biopsies to distinguish these borderline cases.

### Q: "Why not just use the most accurate model?"
**A**: The project goal is **mobile deployment**. VGG16 might achieve higher accuracy but at 138M parameters it's impractical for mobile. EfficientNetB0 (5.3M params) and MobileNetV2 (3.4M params) offer the best accuracy-to-size tradeoff.

### Q: "How does Grad-CAM improve the model?"
**A**: Grad-CAM doesn't improve accuracy — it improves **trust and interpretability**. It verifies the model attends to clinically relevant features (lesion borders, color patterns) rather than dataset artifacts (ruler marks, skin tone).

### Q: "What about real-world deployment concerns?"
**A**: Key considerations:
- Model is a **screening aid**, not a diagnostic tool
- Always requires dermatologist confirmation
- Performance may vary on images taken with different cameras/lighting
- Regulatory compliance (FDA/CE marking) needed for clinical use

---

## Metrics Cheat Sheet

| Metric | What It Measures | When to Use |
|---|---|---|
| **Accuracy** | Overall correctness | ❌ Misleading with imbalanced data |
| **Macro F1** | Average F1 across all classes equally | ✅ Primary metric for imbalanced multi-class |
| **Weighted F1** | F1 weighted by class frequency | ⚠️ Biased toward majority class |
| **Recall** | How many actual positives were caught | ✅ Critical for clinical sensitivity |
| **Precision** | How many predicted positives were correct | ✅ Important for false alarm rate |
| **AUC-ROC** | Discrimination ability across thresholds | ✅ Good for per-class comparison |

---

## Next Steps After Defense

1. [ ] Complete training for all 4 models with oversampling
2. [ ] Create comparative results table
3. [ ] Select best model for mobile deployment
4. [ ] Convert to TFLite format
5. [ ] Build Streamlit/FastAPI inference webapp
6. [ ] Document API endpoints for mobile integration

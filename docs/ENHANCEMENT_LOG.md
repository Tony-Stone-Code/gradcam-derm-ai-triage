# X-Skin Enhancement Log

All changes made to optimize the skin cancer detection CNN pipeline.

---

## Enhancement 1: Baseline Training Scripts
**Date**: 2026-05-12
**Files Created**:
- `training/train_vgg16.py`
- `training/train_resnet50.py`
- `training/train_mobilenetv2.py`
- `training/train_efficientnetb0.py`

**What**: Four self-contained Kaggle-ready training scripts replacing the original monolithic notebook (`CNN_skincancer_FIX.ipynb`).

**Why**: One notebook per model maximizes Kaggle free-tier GPU utilization. Each script includes data loading, training, evaluation, and Grad-CAM.

**Initial Config**: Balanced class weights (raw), focal loss (gamma=2.0), 256-unit head, single-layer fine-tuning.

---

## Enhancement 2: Sqrt-Dampened Class Weights
**Date**: 2026-05-12
**Impact**: EfficientNetB0 accuracy 61% → 79% (+18%)

**Problem**: Raw `balanced` class weights gave DF class ~30× weight, causing massive over-prediction of minority classes. Precision collapsed (MEL: 0.28, VASC: 0.27).

**Fix**: `class_weights = np.sqrt(class_weights)` reduces the weight ratio while still helping minority classes.

**Before** (raw balanced):
```
NV weight:  ~0.15    DF weight: ~30.0
```

**After** (sqrt dampened):
```
NV weight:  ~0.39    DF weight: ~5.5
```

---

## Enhancement 3: Tuned Focal Loss Gamma
**Date**: 2026-05-13
**Impact**: Better precision-recall balance

**Problem**: `gamma=2.0` was too aggressive — over-focused on hard-to-classify samples at the expense of overall accuracy.

**Fix**: Reduced to `gamma=1.5` for a softer focus on hard examples.

---

## Enhancement 4: Deeper Fine-tuning
**Date**: 2026-05-13
**Impact**: Better feature adaptation to dermoscopic images

**Problem**: Conservative fine-tuning (e.g., EfficientNetB0 from layer 200 = only 37 layers unfrozen) limited the model's ability to adapt ImageNet features to skin lesion patterns.

**Fix per architecture**:
| Model | Before | After | Layers Unfrozen |
|---|---|---|---|
| VGG16 | 15 | **11** | ~8 |
| ResNet50 | 143 | **120** | ~55 |
| MobileNetV2 | 100 | **80** | ~75 |
| EfficientNetB0 | 200 | **150** | ~87 |

---

## Enhancement 5: Larger Classification Head
**Date**: 2026-05-13
**Impact**: More capacity to learn class boundaries

**Before**: `GAP → BN → Drop(0.3) → Dense(256) → BN → Drop(0.3) → Dense(7)`

**After**: `GAP → BN → Drop(0.4) → Dense(512) → BN → Drop(0.3) → Dense(128) → BN → Drop(0.2) → Dense(7)`

**Why**: The 7-class problem with visually similar classes needs more parameters in the decision layers.

---

## Enhancement 6: Training Hyperparameter Tuning
**Date**: 2026-05-13

| Parameter | Before | After | Reason |
|---|---|---|---|
| Phase 1 epochs | 10 | **15** | More time to learn head |
| Phase 2 epochs | 20 | **30** | More fine-tuning time |
| Phase 2 LR | 1e-4 | **5e-5** | Prevent catastrophic forgetting |
| EarlyStopping patience | 5 | **7** | Don't stop too early |
| ReduceLR patience | 3 | **4** | More exploration before LR drop |
| ReduceLR min_lr | default | **1e-7** | Hard floor on learning rate |

---

## Enhancement 7: Minority Class Oversampling (KEY FIX)
**Date**: 2026-05-13
**Impact**: Expected to push accuracy past 80% and significantly boost macro F1

**Problem**: Class weights adjust gradient magnitude but the model still only sees ~90 DF images per epoch. With ~5,360 NV images, the model learns 60× more NV diversity.

**Fix**: Duplicate minority class rows to 50% of majority class count before feeding to the augmented data generator.

```python
max_count = train_df['dx'].value_counts().max()    # ~5,360
target_count = max_count // 2                       # ~2,680

for each class:
    if count < target_count:
        repeat rows until target_count reached
```

**Result**: Training set grows from ~8,000 to ~22,000+ samples with near-balanced distribution. Combined with random augmentation, each minority image is seen in many unique augmented forms per epoch.

---

## Enhancement 8: Web Application Landing Page
**Date**: 2026-05-21

**What**: Created a professional Home page for the X-Skin Streamlit web application, providing real-world context for the diagnostic tool.

**Details**:
- Incorporated skin cancer statistics localised to Ghana and West Africa, sourced from the WHO Global Cancer Observatory (GCO)
- Added detailed descriptions of all 7 diagnostic classes (NV, MEL, BKL, BCC, AKIEC, VASC, DF) so that end-users understand what the model is classifying
- Included developer information (Anthony Opoku-Acheampong, Student ID 4245230003, BSc Data Science and Analytics)
- Mobile-responsive layout with sidebar navigation between the Home page and the Diagnosis page

---

## Enhancement 9: Lazy Loading Architecture
**Date**: 2026-05-22
**Impact**: Eliminates cloud deployment timeouts on Streamlit Community Cloud and HuggingFace Spaces

**Problem**: Loading all CNN model weights at server boot caused the application to exceed the health-check timeout window on free-tier cloud platforms. The server was killed before it could begin serving requests.

**Fix**: Refactored `InferenceEngine` to use **deferred model initialisation**. Models are loaded into memory only when a user first requests inference for a given architecture, not at application startup.

**Implementation**:
- Introduced a `_get_or_load_model()` method that checks a dictionary-based cache before loading
- On the first inference request for a model, the method loads the weights from disk, caches the model object, and returns it
- Subsequent requests for the same model retrieve it from the cache with zero additional load time
- Server boot time reduced from ~30–60 seconds (all models) to < 2 seconds (no models pre-loaded)

---

## Enhancement 10: Out-of-Distribution (OOD) Rejection Gate
**Date**: 2026-05-25
**Impact**: Prevents misleading diagnoses on non-dermatoscopic input images

**Problem**: The softmax output layer always distributes exactly 100% probability across the 7 classes, regardless of whether the input image is a skin lesion or an unrelated photograph (e.g., a landscape, a pet, a selfie). This means the model will always return a diagnosis, even when the input is completely outside its training domain.

**Fix**: Implemented a **dual-gate validation** mechanism in `app.py` via the `is_valid_prediction()` function. Both gates must pass for the prediction to be displayed:

| Gate | Condition | Threshold | Rationale |
|---|---|---|---|
| Confidence | `max(probabilities) >= 0.50` | 50% | A genuine skin lesion should produce at least moderate confidence in one class |
| Shannon Entropy | `H(p) <= 1.6` | 1.6 nats | Uniform distribution over 7 classes gives `H = log(7) ≈ 1.946`; threshold of 1.6 provides margin |

**Result**: Non-skin images now receive a clear rejection message ("Not a Valid Dermatoscopic Skin Lesion") instead of a spurious diagnosis.

---

## Results Tracker

| Model | Version | Accuracy | Macro F1 | MEL Recall | Status |
|---|---|---|---|---|---|
| EfficientNetB0 | v1 (raw weights) | 61.1% | 0.5147 | 0.59 | ❌ Over-predicting minorities |
| EfficientNetB0 | v2 (enhanced) | **78.8%** | **0.7044** | **0.65** | ✅ Good balance |
| MobileNetV2 | v1 (sqrt weights only) | 78.0% | 0.5471 | 0.37 | ⚠️ Low minority recall |
| MobileNetV2 | v2 (enhanced+oversample) | — | — | — | 🔄 Pending |
| VGG16 | v2 (enhanced+oversample) | — | — | — | 🔄 Pending |
| ResNet50 | v2 (enhanced+oversample) | — | — | — | 🔄 Pending |

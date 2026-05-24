import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMG_SIZE = (224, 224)

class InferenceEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}  # Cache for loaded models
        self.meta_model = None
        self.ensemble_weights = None

    def get_available_models(self):
        """Scans the models_dir for available models without loading them into RAM."""
        available_models = []
        has_rf = False
        
        # Check for stacking ensemble
        rf_path = os.path.join(self.models_dir, 'stacking_ensemble', 'meta_rf.pkl')
        if os.path.exists(rf_path):
            has_rf = True

        if not os.path.exists(self.models_dir):
            return available_models, has_rf

        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('.keras') or model_file.endswith('.h5'):
                model_name = os.path.splitext(model_file)[0]
                available_models.append(model_name)
                    
        return available_models, has_rf

    def _get_or_load_model(self, model_name):
        """Lazily loads a model into memory only when requested to prevent boot timeouts."""
        if model_name not in self.models:
            print(f"Loading {model_name} into memory...")
            path = os.path.join(self.models_dir, f"{model_name}.keras")
            if not os.path.exists(path):
                path = os.path.join(self.models_dir, f"{model_name}.h5")
            self.models[model_name] = tf.keras.models.load_model(path, compile=False)
        return self.models[model_name]

    def _get_or_load_meta_model(self):
        if self.meta_model is None:
            rf_path = os.path.join(self.models_dir, 'stacking_ensemble', 'meta_rf.pkl')
            self.meta_model = joblib.load(rf_path)
        return self.meta_model

    def preprocess_image(self, img, model_name):
        """Preprocesses PIL Image depending on the model."""
        img_resized = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_resized)
        raw_array = np.copy(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if 'resnet' in model_name.lower():
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        elif 'vgg' in model_name.lower():
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        elif 'mobilenet' in model_name.lower():
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        elif 'efficientnet' in model_name.lower():
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        else:
            img_array = img_array / 255.0

        return img_array, raw_array

    def predict_single_model(self, img, model_name):
        processed_img, raw_array = self.preprocess_image(img, model_name)
        model = self._get_or_load_model(model_name)
        preds = model.predict(processed_img, verbose=0)
        return preds[0], raw_array

    def predict_weighted_voting(self, img):
        if self.ensemble_weights is None:
            weight_path = os.path.join(self.models_dir, 'decoupled_ensemble', 'ensemble_weights.json')
            if os.path.exists(weight_path):
                with open(weight_path, 'r') as f:
                    self.ensemble_weights = json.load(f)
            else:
                available, _ = self.get_available_models()
                self.ensemble_weights = {m: 1.0/len(available) for m in available}

        final_probs = np.zeros(len(CLASSES))
        for model_name, weight in self.ensemble_weights.items():
            probs, _ = self.predict_single_model(img, model_name)
            final_probs += probs * weight
            
        return final_probs / np.sum(list(self.ensemble_weights.values()))

    def predict_rank_based(self, img):
        available_models, _ = self.get_available_models()
        num_classes = len(CLASSES)
        borda_scores = np.zeros(num_classes)
        
        for model_name in available_models:
            probs, _ = self.predict_single_model(img, model_name)
            ranked_indices = np.argsort(probs)
            for rank, class_idx in enumerate(ranked_indices):
                borda_scores[class_idx] += rank
                
        final_probs = borda_scores / np.sum(borda_scores)
        return final_probs

    def predict_stacking(self, img):
        available_models, _ = self.get_available_models()
        meta_features = []
        for model_name in sorted(available_models):
            probs, _ = self.predict_single_model(img, model_name)
            meta_features.extend(probs)
            
        meta_features = np.array(meta_features).reshape(1, -1)
        meta_model = self._get_or_load_meta_model()
        
        if hasattr(meta_model, 'predict_proba'):
            return meta_model.predict_proba(meta_features)[0]
        else:
            pred_idx = int(meta_model.predict(meta_features)[0])
            one_hot = np.zeros(len(CLASSES))
            one_hot[pred_idx] = 1.0
            return one_hot

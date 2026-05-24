import os
from PIL import Image
from inference import InferenceEngine
from xai_utils import make_gradcam_heatmap

# Create a dummy image
img = Image.new('RGB', (224, 224), color = 'red')

engine = InferenceEngine(models_dir="models")
engine.load_models()

print("Loaded models:", engine.available_architectures)

if engine.available_architectures:
    model_name = engine.available_architectures[0]
    processed_img, raw_array = engine.preprocess_image(img, model_name)
    xai_model = engine.models[model_name]
    
    print("Running Grad-CAM...")
    try:
        heatmap = make_gradcam_heatmap(processed_img, xai_model)
        print("Success! Heatmap shape:", heatmap.shape)
    except Exception as e:
        import traceback
        traceback.print_exc()

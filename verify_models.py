import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def verify_models():
    print("=" * 60)
    print("FOREST GUARDIAN AI — MODEL VERIFICATION SUITE")
    print("=" * 60)

    # 1. Models to verify
    models_to_check = [
        {
            "name": "Custom CNN",
            "file": "custom_cnn_model.h5",
            "preprocess_type": "internal_rescaling" # Model contains Rescaling(1./255)
        },
        {
            "name": "MobileNetV2",
            "file": "mobilenet_model.h5",
            "preprocess_type": "mobilenet_v2" # mobilenet_v2.preprocess_input
        },
        {
            "name": "ResNet50",
            "file": "resnet_model.h5",
            "preprocess_type": "resnet50" # resnet50.preprocess_input
        }
    ]

    loaded_models = {}

    # Step 1: Load all 3 models
    print("\nSTEP 1: LOADING MODELS")
    print("-" * 60)
    for item in models_to_check:
        filepath = item["file"]
        model_name = item["name"]
        
        if not os.path.exists(filepath):
            print(f"ERROR: Model file '{filepath}' not found!")
            sys.exit(1)
            
        try:
            print(f"Loading {model_name} from '{filepath}'...")
            model = load_model(filepath)
            loaded_models[model_name] = model
            
            # Print details
            input_shape = model.input_shape
            output_shape = model.output_shape
            total_params = model.count_params()
            
            print(f"  • Model Name:       {model.name or model_name}")
            print(f"  • Input Shape:      {input_shape}")
            print(f"  • Output Shape:     {output_shape}")
            print(f"  • Total Parameters: {total_params:,}")
            print("  Status: LOADED SUCCESSFULLY\n")
        except Exception as e:
            print(f"ERROR: Failed to load {model_name} from '{filepath}': {str(e)}")
            sys.exit(1)

    # Step 2: Select test images for verification
    test_dir = os.path.join("processed_data", "test")
    fire_dir = os.path.join(test_dir, "Fire")
    non_fire_dir = os.path.join(test_dir, "Non-Fire")

    if not os.path.exists(fire_dir) or not os.path.exists(non_fire_dir):
        print("ERROR: Test dataset directories not found in 'processed_data/test/'.")
        sys.exit(1)

    fire_images = [os.path.join(fire_dir, f) for f in sorted(os.listdir(fire_dir))[:2]]
    non_fire_images = [os.path.join(non_fire_dir, f) for f in sorted(os.listdir(non_fire_dir))[:2]]

    sample_images = [
        (img_path, "Fire") for img_path in fire_images
    ] + [
        (img_path, "Non-Fire") for img_path in non_fire_images
    ]

    print("\nSTEP 2: RUNNING INFERENCE ON SAMPLE TEST IMAGES")
    print("-" * 60)

    for img_path, ground_truth in sample_images:
        filename = os.path.basename(img_path)
        print(f"\nImage: {filename} (Expected Ground Truth: {ground_truth})")
        print("=" * 60)

        # Load raw image 224x224x3 RGB (0-255 range)
        raw_img = load_img(img_path, target_size=(224, 224), color_mode="rgb")
        raw_array = img_to_array(raw_img) # float32 array in [0, 255]

        for item in models_to_check:
            m_name = item["name"]
            p_type = item["preprocess_type"]
            model = loaded_models[m_name]

            # Prepare image array based on model preprocessing requirement
            if p_type == "internal_rescaling":
                # Custom CNN expects [0, 255] RGB float array (Rescaling layer inside model)
                input_tensor = np.expand_dims(raw_array.copy(), axis=0)
            elif p_type == "mobilenet_v2":
                # External MobileNetV2 preprocessing
                input_tensor = np.expand_dims(mobilenet_preprocess(raw_array.copy()), axis=0)
            elif p_type == "resnet50":
                # External ResNet50 preprocessing
                input_tensor = np.expand_dims(resnet_preprocess(raw_array.copy()), axis=0)
            else:
                input_tensor = np.expand_dims(raw_array.copy(), axis=0)

            try:
                # Predict
                prob = float(model.predict(input_tensor, verbose=0)[0][0])
                
                # Sigmoid output represents P(Non-Fire)
                # prob < 0.5 -> Fire (0), prob >= 0.5 -> Non-Fire (1)
                if prob < 0.5:
                    pred_class = "Fire"
                    confidence = (1.0 - prob) * 100.0
                else:
                    pred_class = "Non-Fire"
                    confidence = prob * 100.0

                print(f"  Model: {m_name:<15} | Pred: {pred_class:<8} | Sigmoid P(Non-Fire): {prob:.6f} | Confidence: {confidence:6.2f}%")

            except Exception as e:
                print(f"ERROR: Inference failed for model '{m_name}' on image '{filename}': {str(e)}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("MODEL VERIFICATION: PASSED")
    print("=" * 60)

if __name__ == "__main__":
    verify_models()

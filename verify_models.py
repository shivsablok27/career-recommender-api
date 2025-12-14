import joblib
import numpy as np
import os
import sys

# Paths
PERSONALITY_MODEL_PATH = r"d:/MLFYP_UNSUPERVISED/RIASECOC2E2A2N clustering/career_kmeans_model.joblib"
READING_MODEL_PATH = r"d:/MLFYP_UNSUPERVISED/ReadingCompAnalysis/reading_kmeans_model.joblib"

def verify_model(name, path):
    print(f"--- Verifying {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        return False
    
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully: {type(model)}")
        
        if hasattr(model, 'cluster_centers_'):
            print(f"Cluster centers shape: {model.cluster_centers_.shape}")
        else:
            print("WARNING: No cluster_centers_ attribute found.")

        if hasattr(model, 'transform'):
            print("Transform method available (needed for soft voting).")
            # Create dummy input to test transform
            n_features = model.cluster_centers_.shape[1]
            dummy_input = np.random.rand(1, n_features)
            distances = model.transform(dummy_input)
            print(f"Dummy transform output shape: {distances.shape}")
            print(f"Distances: {distances}")
        else:
            print("ERROR: Transform method NOT available.")
            return False
            
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

def main():
    p_ok = verify_model("Personality Model", PERSONALITY_MODEL_PATH)
    print("\n")
    r_ok = verify_model("Reading Model", READING_MODEL_PATH)
    
    if p_ok and r_ok:
        print("\nSUCCESS: Both models verified.")
    else:
        print("\nFAILURE: One or more models failed verification.")

if __name__ == "__main__":
    main()

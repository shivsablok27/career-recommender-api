import joblib
import numpy as np
import os
import pandas as pd

# --- Constants ---
# Use relative paths for portability (works on any machine/server)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PERSONALITY_MODEL_PATH = os.path.join(MODELS_DIR, "career_kmeans_model.joblib")
PERSONALITY_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
PERSONALITY_MAPPING_PATH = os.path.join(MODELS_DIR, "job_mapping.pkl")

READING_MODEL_PATH = os.path.join(MODELS_DIR, "reading_kmeans_model.joblib")
READING_SCALER_PATH = os.path.join(MODELS_DIR, "reading_scaler.joblib")

# Feature Order
PERSONALITY_FEATURES = ['R', 'I', 'A', 'S', 'E', 'C', 'O', 'C2', 'E2', 'A2', 'N']
READING_FEATURES = ['pattern_focus', 'problem_solving_focus', 'management_design_focus', 
                    'attention_consistency', 'preference_alignment', 'engagement_score']

# Domain Standardization
DOMAIN_DS = 'Data Science'
DOMAIN_SD = 'Software Development'
DOMAIN_TPM = 'Tech Project Management'

# Mapping from internal labels to standardized domains
# Personality model uses specific keys, we map them here
PERSONALITY_DOMAIN_MAP = {
    'Data_Science_AI_ML': DOMAIN_DS,
    'Software_Developer': DOMAIN_SD,
    'Tech_Project_Manager': DOMAIN_TPM
}

class ModelInference:
    def __init__(self):
        self.p_model = None
        self.p_scaler = None
        self.p_mapping = None
        self.r_model = None
        self.r_scaler = None
        self.r_mapping = None # Dict mapping cluster_idx -> Domain (or 'Mixed')

    def load_models(self):
        """Loads all models, scalers, and mappings."""
        print("Loading Personality Model...")
        self.p_model = joblib.load(PERSONALITY_MODEL_PATH)
        self.p_scaler = joblib.load(PERSONALITY_SCALER_PATH)
        self.p_mapping = joblib.load(PERSONALITY_MAPPING_PATH)
        
        print("Loading Reading Model...")
        self.r_model = joblib.load(READING_MODEL_PATH)
        self.r_scaler = joblib.load(READING_SCALER_PATH)
        self._derive_reading_mapping()
        
        print("Models loaded successfully.")

    def _derive_reading_mapping(self):
        """
        Dynamically maps reading clusters to domains based on centroids.
        Cluster with max 'pattern_focus' -> Data Science
        Cluster with max 'problem_solving_focus' -> Software Dev
        Cluster with max 'management_design_focus' -> Tech Project Mgmt
        Remaining -> Mixed (Ignored in final prediction)
        """
        centroids = self.r_model.cluster_centers_
        # Feature indices: 0=pattern, 1=problem, 2=mgmt
        
        self.r_mapping = {}
        assigned_clusters = set()
        
        # 1. Find Analytical (Data Science) - Max Pattern Focus
        idx_ds = np.argmax(centroids[:, 0])
        self.r_mapping[idx_ds] = DOMAIN_DS
        assigned_clusters.add(idx_ds)
        
        # 2. Find Technical (Software Dev) - Max Problem Solving (excluding already assigned)
        remaining = [i for i in range(4) if i not in assigned_clusters]
        best_sd = -1
        max_val = -1
        for i in remaining:
            if centroids[i, 1] > max_val:
                max_val = centroids[i, 1]
                best_sd = i
        self.r_mapping[best_sd] = DOMAIN_SD
        assigned_clusters.add(best_sd)
        
        # 3. Find Managerial (TPM) - Max Mgmt Focus
        remaining = [i for i in range(4) if i not in assigned_clusters]
        best_tpm = -1
        max_val = -1
        for i in remaining:
            if centroids[i, 2] > max_val:
                max_val = centroids[i, 2]
                best_tpm = i
        self.r_mapping[best_tpm] = DOMAIN_TPM
        assigned_clusters.add(best_tpm)
        
        # 4. Mixed (Ignored)
        remaining = [i for i in range(4) if i not in assigned_clusters]
        if remaining:
            self.r_mapping[remaining[0]] = 'Mixed'
            
        print(f"Derived Reading Mapping: {self.r_mapping}")

    def _distances_to_probs(self, distances):
        """
        Converts distances to probabilities using Softmax of negative distances.
        """
        # Negate distances so closer = higher value
        neg_dists = -distances
        # Softmax
        exp_dists = np.exp(neg_dists - np.max(neg_dists)) # Shift for stability
        return exp_dists / exp_dists.sum()

    def predict_personality(self, features_dict):
        """
        Predicts domain probabilities based on personality traits.
        Args:
            features_dict: Dict with keys ['R', 'I', 'A', 'S', 'E', 'C', 'O', 'C2', 'E2', 'A2', 'N']
        Returns:
            Dict {Domain: Probability}
        """
        # Prepare input vector
        vector = np.array([[features_dict.get(f, 0.5) for f in PERSONALITY_FEATURES]])
        
        # Scale
        scaled_vector = self.p_scaler.transform(vector)
        
        # Get distances
        distances = self.p_model.transform(scaled_vector)[0]
        
        # Convert to probs
        probs = self._distances_to_probs(distances)
        
        # Map to domains
        result = {DOMAIN_DS: 0.0, DOMAIN_SD: 0.0, DOMAIN_TPM: 0.0}
        
        for cluster_idx, prob in enumerate(probs):
            # p_mapping maps cluster_idx -> internal_label
            internal_label = self.p_mapping.get(cluster_idx)
            # Map internal_label -> standardized domain
            std_domain = PERSONALITY_DOMAIN_MAP.get(internal_label)
            if std_domain:
                result[std_domain] += prob
                
        return result

    def _calculate_reading_features(self, survey_data):
        """
        Transforms raw survey data into cognitive features.
        Args:
            survey_data: Dict with keys:
                - Qpattern1, Qpattern2 (1-5)
                - Qprobsolve1, Qprobsolve2 (1-5)
                - Qmgmt1, Qmgmt2 (1-5)
                - chosenActivity (String)
                - freeText (String)
        Returns:
            Dict of calculated features (0-1 scale)
        """
        # 1. Normalize Likert Scales (1-5 -> 0-1)
        # Formula: (val - 1) / 4
        likert_map = {
            'Qpattern1': survey_data.get('Qpattern1', 3),
            'Qpattern2': survey_data.get('Qpattern2', 3),
            'Qprobsolve1': survey_data.get('Qprobsolve1', 3),
            'Qprobsolve2': survey_data.get('Qprobsolve2', 3),
            'Qmgmt1': survey_data.get('Qmgmt1', 3),
            'Qmgmt2': survey_data.get('Qmgmt2', 3)
        }
        
        norm_scores = {k: (v - 1) / 4.0 for k, v in likert_map.items()}
        
        # 2. Calculate Focus Scores (Mean of pairs)
        pattern_focus = (norm_scores['Qpattern1'] + norm_scores['Qpattern2']) / 2.0
        problem_solving_focus = (norm_scores['Qprobsolve1'] + norm_scores['Qprobsolve2']) / 2.0
        management_design_focus = (norm_scores['Qmgmt1'] + norm_scores['Qmgmt2']) / 2.0
        
        # 3. Calculate Attention Consistency (1 - std_dev)
        focus_scores = [pattern_focus, problem_solving_focus, management_design_focus]
        attention_consistency = 1.0 - np.std(focus_scores)
        
        # 4. Calculate Preference Alignment
        chosen_activity = survey_data.get('chosenActivity', '')
        if 'Discovering hidden trends' in chosen_activity:
            preference_alignment = pattern_focus
        elif 'Fixing a broken system' in chosen_activity:
            preference_alignment = problem_solving_focus
        else: # Management/Design
            preference_alignment = management_design_focus
            
        # 5. Calculate Engagement Score (Text Length)
        # Note: In original notebook, max length was dynamic. Here we assume a reasonable max (e.g., 200 chars)
        # or use a fixed scaling factor. Let's use a sigmoid-like or capped linear scale.
        text_len = len(str(survey_data.get('freeText', '')))
        # Assuming max length in training was around 200-300 chars based on typical short answers
        # We'll cap at 1.0 for 200 chars
        engagement_score = min(text_len / 200.0, 1.0)
        
        return {
            'pattern_focus': pattern_focus,
            'problem_solving_focus': problem_solving_focus,
            'management_design_focus': management_design_focus,
            'attention_consistency': attention_consistency,
            'preference_alignment': preference_alignment,
            'engagement_score': engagement_score
        }

    def predict_reading(self, survey_data):
        """
        Predicts domain probabilities based on raw survey data.
        Args:
            survey_data: Dict of raw survey responses.
        Returns:
            Dict {Domain: Probability}
        """
        # Feature Engineering
        features_dict = self._calculate_reading_features(survey_data)
        
        # Prepare input vector
        vector = np.array([[features_dict.get(f, 0.5) for f in READING_FEATURES]])
        
        # Scale
        scaled_vector = self.r_scaler.transform(vector)
        
        # Get distances
        distances = self.r_model.transform(scaled_vector)[0]
        
        # Convert to probs
        probs = self._distances_to_probs(distances)
        
        # Map to domains
        result = {DOMAIN_DS: 0.0, DOMAIN_SD: 0.0, DOMAIN_TPM: 0.0}
        
        for cluster_idx, prob in enumerate(probs):
            domain = self.r_mapping.get(cluster_idx)
            if domain == 'Mixed':
                # Ignore Mixed cluster as requested
                continue
            elif domain in result:
                result[domain] += prob
        
        # Re-normalize probabilities since we dropped 'Mixed'
        total_prob = sum(result.values())
        if total_prob > 0:
            for k in result:
                result[k] /= total_prob
        
        # --- ALIGNMENT BOOSTING (Post-Processing) ---
        # Boost Factor: 0.15 (Significant but not overwhelming)
        BOOST_FACTOR = 0.15
        
        # 1. Map chosenProject
        project_choice = survey_data.get('chosenProject', '')
        if 'Analyzing data' in project_choice:
            result[DOMAIN_DS] += BOOST_FACTOR
        elif 'Building backend' in project_choice:
            result[DOMAIN_SD] += BOOST_FACTOR
        elif 'Designing user interfaces' in project_choice:
            result[DOMAIN_TPM] += BOOST_FACTOR
            
        # 2. Map team_choice
        team_choice = survey_data.get('team_choice', '')
        if 'Observation and analysis' in team_choice:
            result[DOMAIN_DS] += BOOST_FACTOR
        elif 'Technical troubleshooting' in team_choice:
            result[DOMAIN_SD] += BOOST_FACTOR
        elif 'Design and presentation' in team_choice:
            result[DOMAIN_TPM] += BOOST_FACTOR
            
        # 3. Re-normalize again after boosting
        total_prob_boosted = sum(result.values())
        if total_prob_boosted > 0:
            for k in result:
                result[k] /= total_prob_boosted
                
        return result

if __name__ == "__main__":
    # Simple test
    inference = ModelInference()
    inference.load_models()
    
    # Dummy Personality Data
    dummy_p = {k: 0.8 if k in ['R', 'I'] else 0.2 for k in PERSONALITY_FEATURES}
    print("\nPersonality Preds:", inference.predict_personality(dummy_p))
    
    # Dummy Reading Data (Raw Survey)
    dummy_r_raw = {
        'Qpattern1': 5, 'Qpattern2': 5, # High Pattern
        'Qprobsolve1': 2, 'Qprobsolve2': 2,
        'Qmgmt1': 1, 'Qmgmt2': 1,
        'chosenActivity': 'Discovering hidden trends',
        'freeText': 'I really enjoyed reading about the patterns in the text.' * 5
    }
    print("Reading Preds (Raw Input):", inference.predict_reading(dummy_r_raw))

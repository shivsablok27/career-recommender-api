from model_inference import ModelInference, DOMAIN_DS, DOMAIN_SD, DOMAIN_TPM

class EnsemblePredictor:
    def __init__(self):
        self.inference = ModelInference()
        self.inference.load_models()

    def predict(self, personality_data, reading_data, weights=(0.5, 0.5)):
        """
        Combines predictions from both models.
        Args:
            personality_data: Dict of personality traits.
            reading_data: Dict of reading features.
            weights: Tuple (weight_personality, weight_reading).
        Returns:
            Dict containing:
            - 'final_recommendation': The top domain.
            - 'final_scores': Dict of combined scores.
            - 'personality_scores': Dict of personality model scores.
            - 'reading_scores': Dict of reading model scores.
        """
        # Get individual model predictions
        p_scores = self.inference.predict_personality(personality_data)
        r_scores = self.inference.predict_reading(reading_data)
        
        # Combine scores
        w_p, w_r = weights
        final_scores = {}
        
        for domain in [DOMAIN_DS, DOMAIN_SD, DOMAIN_TPM]:
            # Weighted Average
            score = (w_p * p_scores.get(domain, 0)) + (w_r * r_scores.get(domain, 0))
            final_scores[domain] = score
            
        # Determine winner
        # Sort by score descending
        sorted_domains = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        recommendation = sorted_domains[0][0]
        
        return {
            'final_recommendation': recommendation,
            'final_scores': final_scores,
            'personality_scores': p_scores,
            'reading_scores': r_scores
        }

if __name__ == "__main__":
    predictor = EnsemblePredictor()
    
    # Test Case 1: Strong Software Dev (Both models agree)
    print("\n--- Test Case 1: The Coder ---")
    p_data = {'R': 0.9, 'I': 0.7, 'A': 0.3, 'S': 0.2, 'E': 0.3, 'C': 0.6, 'O': 0.5, 'C2': 0.6, 'E2': 0.3, 'A2': 0.5, 'N': 0.2}
    r_data = {'problem_solving_focus': 0.9, 'pattern_focus': 0.4, 'management_design_focus': 0.2, 'attention_consistency': 0.8, 'preference_alignment': 0.9, 'engagement_score': 0.7}
    
    result = predictor.predict(p_data, r_data)
    print(f"Recommendation: {result['final_recommendation']}")
    print(f"Final Scores: {result['final_scores']}")
    
    # Test Case 2: Conflicting (Personality=TPM, Reading=DS)
    print("\n--- Test Case 2: The Conflicted Manager ---")
    p_data = {'R': 0.2, 'I': 0.3, 'A': 0.3, 'S': 0.9, 'E': 0.9, 'C': 0.5, 'O': 0.6, 'C2': 0.5, 'E2': 0.9, 'A2': 0.8, 'N': 0.2} # High S, E -> TPM
    r_data = {'pattern_focus': 0.9, 'problem_solving_focus': 0.3, 'management_design_focus': 0.2, 'attention_consistency': 0.8, 'preference_alignment': 0.9, 'engagement_score': 0.7} # High Pattern -> DS
    
    result = predictor.predict(p_data, r_data)
    print(f"Recommendation: {result['final_recommendation']}")
    print(f"Final Scores: {result['final_scores']}")
    print(f"Personality Scores: {result['personality_scores']}")
    print(f"Reading Scores: {result['reading_scores']}")

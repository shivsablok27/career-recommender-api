from ensemble_predictor import EnsemblePredictor

def main():
    # ==========================================
    #           PUT YOUR DATA HERE
    # ==========================================
    
    # Personality Traits (Scale 0.0 to 1.0)
    # R: Realistic, I: Investigative, A: Artistic, S: Social, E: Enterprising, C: Conventional
    # O: Openness, C2: Conscientiousness, E2: Extraversion, A2: Agreeableness, N: Neuroticism
    # Personality Traits (Data Scientist Profile - O*NET inspired)
    # High Investigative (I), High Openness (O), High Conscientiousness (C2)
    my_personality_data = {
        'R': 0.60, # Realistic: Moderate (Working with data/tools)
        'I': 0.95, # Investigative: Very High (Research, Logic)
        'A': 0.40, # Artistic: Low-Moderate
        'S': 0.30, # Social: Low
        'E': 0.40, # Enterprising: Low-Moderate
        'C': 0.80, # Conventional: High (Structured data, accuracy)
        'O': 0.90, # Openness: High (Curiosity, new methods)
        'C2': 0.85, # Conscientiousness: High (Detail-oriented)
        'E2': 0.40, # Extraversion: Low-Moderate
        'A2': 0.50, # Agreeableness: Moderate
        'N': 0.30  # Neuroticism: Low
    }

    # Reading Metrics (Raw Survey Data)
    # Likert Scale: 1 (Strongly Disagree) to 5 (Strongly Agree)
    my_reading_data = {
        'Qpattern1': 5, # I enjoy finding patterns...
        'Qpattern2': 5, # I look for hidden trends...
        'Qprobsolve1': 3, # I like fixing broken things...
        'Qprobsolve2': 3, # I enjoy troubleshooting...
        'Qmgmt1': 2, # I like managing people...
        'Qmgmt2': 2, # I enjoy designing workflows...
        
        # Options: 
        # 'Discovering hidden trends or logical patterns' (Analytical)
        # 'Fixing a broken system or solving a technical problem' (Technical)
        # 'Managing a team or designing a process' (Managerial)
        'chosenActivity': 'Discovering hidden trends or logical patterns',
        
        # Free text response about their reading experience
        'freeText': 'I really enjoy finding patterns and discovering hidden trends in data. Analyzing complex information to reveal underlying structures is fascinating to me. I love using logic to solve problems and predict future outcomes based on evidence.'
    }
    
    # ==========================================
    #        END OF DATA ENTRY
    # ==========================================

    print("Initializing Ensemble Predictor...")
    predictor = EnsemblePredictor()
    
    print("\nProcessing your data...")
    result = predictor.predict(my_personality_data, my_reading_data)
    
    print("\n" + "="*40)
    print(f"FINAL RECOMMENDATION: {result['final_recommendation']}")
    print("="*40)
    
    print("\nDetailed Scores:")
    for domain, score in result['final_scores'].items():
        print(f"  {domain}: {score:.4f}")
        
    print("\nModel Breakdown:")
    print("  Personality Model Contribution:")
    for domain, score in result['personality_scores'].items():
        print(f"    {domain}: {score:.4f}")
        
    print("  Reading Model Contribution:")
    for domain, score in result['reading_scores'].items():
        print(f"    {domain}: {score:.4f}")

if __name__ == "__main__":
    main()

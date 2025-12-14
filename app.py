import streamlit as st
import pandas as pd
import numpy as np
from ensemble_predictor import EnsemblePredictor

# Page Config
st.set_page_config(
    page_title="Career Recommender System",
    page_icon="🎓",
    layout="wide"
)

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return EnsemblePredictor()

predictor = get_predictor()

# --- Header ---
st.title("🎓 AI Career Recommender System")
st.markdown("""
This system analyzes your **Personality Traits** and **Cognitive Reading Style** to recommend the best fit career path among:
*   **Data Science / AI / ML**
*   **Software Development**
*   **Tech Project Management**
""")

st.divider()

# --- Sidebar: Personality Assessment ---
st.sidebar.header("🧠 Personality Assessment")
st.sidebar.info("Rate yourself on the following traits (0 = Low, 1 = High)")

# RIASEC
st.sidebar.subheader("RIASEC Traits")
r_score = st.sidebar.slider("Realistic (Doers)", 0.0, 1.0, 0.5, help="Practical, hands-on, tool-oriented")
i_score = st.sidebar.slider("Investigative (Thinkers)", 0.0, 1.0, 0.5, help="Analytical, intellectual, scientific")
a_score = st.sidebar.slider("Artistic (Creators)", 0.0, 1.0, 0.5, help="Creative, original, independent")
s_score = st.sidebar.slider("Social (Helpers)", 0.0, 1.0, 0.5, help="Cooperative, supporting, teaching")
e_score = st.sidebar.slider("Enterprising (Persuaders)", 0.0, 1.0, 0.5, help="Competitive, leadership, selling")
c_score = st.sidebar.slider("Conventional (Organizers)", 0.0, 1.0, 0.5, help="Detail-oriented, structured, clerical")

# OCEAN
st.sidebar.subheader("Big 5 Traits (OCEAN)")
o_score = st.sidebar.slider("Openness", 0.0, 1.0, 0.5, help="Curious, imaginative, open to new experiences")
c2_score = st.sidebar.slider("Conscientiousness", 0.0, 1.0, 0.5, help="Organized, dependable, disciplined")
e2_score = st.sidebar.slider("Extraversion", 0.0, 1.0, 0.5, help="Outgoing, energetic, social")
a2_score = st.sidebar.slider("Agreeableness", 0.0, 1.0, 0.5, help="Compassionate, cooperative, trusting")
n_score = st.sidebar.slider("Neuroticism", 0.0, 1.0, 0.5, help="Sensitive, nervous, prone to worry")

personality_data = {
    'R': r_score, 'I': i_score, 'A': a_score, 'S': s_score, 'E': e_score, 'C': c_score,
    'O': o_score, 'C2': c2_score, 'E2': e2_score, 'A2': a2_score, 'N': n_score
}

# --- Main Area: Reading Comprehension ---
st.header("📖 Reading Comprehension Assessment")
st.info("Please read the passage below and answer the questions HONESTLY, AS PER YOUR OWN PERSONALITY.")

# Reading Passage
with st.expander("📄 Read the Passage (Click to Expand)", expanded=True):
    st.markdown("""
    **Passage:**
    
    During a college cultural fest, the organizing committee split into smaller groups to manage different parts of the event. 
    One group spent most of their time quietly observing how people moved around the venue—why certain stalls drew larger crowds, 
    why some activities became popular at specific hours, and what patterns explained these changes. 
    
    Another group focused on the technical setup, dealing with sound systems, projectors, and registration devices. 
    Whenever something stopped working, they immediately came together to test what had gone wrong and fix it before the next performance. 
    
    The third group was mostly concerned with the look and feel of the event. They arranged decorations, adjusted lighting, 
    designed banners, and made sure every booth looked appealing and easy to navigate. 
    
    By the end of the day, even though everyone had been part of the same fest, each group felt connected to very different parts of the experience—observing patterns, solving problems, or shaping presentation.
    """)

st.subheader("Survey Questions")
st.markdown("*Please select the option that best describes you (1 = Strongly Disagree, 5 = Strongly Agree)*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Set 1")
    q1 = st.slider("Q1. I found myself focusing on patterns or behaviours, such as why some stalls became more crowded than others.", 1, 5, 3)
    q2 = st.slider("Q2. I found myself focusing on technical issues or devices, imagining how I would fix them.", 1, 5, 3)
    q3 = st.slider("Q3. I found myself focusing on the design, appearance, or arrangement of the event.", 1, 5, 3)

with col2:
    st.markdown("##### Set 2")
    q4 = st.slider("Q4. I enjoy understanding why something happens by observing behaviour or patterns over time.", 1, 5, 3)
    q5 = st.slider("Q5. When something stops working, my first instinct is to explore the system and figure out how to fix it.", 1, 5, 3)
    q6 = st.slider("Q6. I care a lot about how things look, and how people experience them visually or interactively.", 1, 5, 3)

st.divider()

col3, col4 = st.columns(2)

with col3:
    chosen_activity = st.selectbox(
        "Which activity feels the most satisfying to you?",
        [
            "Discovering hidden trends or logical patterns",
            "Fixing a broken system or solving a technical problem",
            "Designing or improving the visual and user experience"
        ]
    )
    
    team_choice = st.selectbox(
        "If you were part of the event described above, which team would you naturally join?",
        [
            "Observation and analysis team",
            "Technical troubleshooting team",
            "Design and presentation team"
        ]
    )

with col4:
    chosen_project = st.selectbox(
        "If you were given a project in college, which task would you enjoy the most?",
        [
            "Analyzing data to understand behaviour or predict outcomes",
            "Building backend logic or solving programming/technical issues",
            "Designing user interfaces and ensuring smooth interaction"
        ]
    )
    
    career_align = st.selectbox(
        "Which career aligns the most with you?",
        [
            "Software Development Engineer (SDE)",
            "Full-Stack Developer",
            "Data Science / Machine Learning"
        ]
    )

st.markdown("##### Reflection")
st.markdown('"IT SHOULD BE A TECHNICAL RESPONSE LIKE A CS/IT STUDENT, NOT A NON TECH TYPE RESPONSE"')
free_text = st.text_area(
    "In a few sentences, explain which part of the event you personally noticed the most, and which you feel you would be most suited for in the event and why it stood out to you.",
    height=150,
    placeholder="Describe it in your own words, focusing on your thoughts and what you found meaningful..."
)

# Mapping inputs to model expected keys
reading_data = {
    'Qpattern1': q1,
    'Qpattern2': q4,
    'Qprobsolve1': q2,
    'Qprobsolve2': q5,
    'Qmgmt1': q3,
    'Qmgmt2': q6,
    'chosenActivity': chosen_activity,
    'freeText': free_text,
    'chosenProject': chosen_project,
    'team_choice': team_choice
    # career_align is collected but not used by the model currently
}

# --- Prediction ---
st.divider()
if st.button("🚀 Generate Career Recommendation", type="primary"):
    if len(free_text) < 10:
        st.warning("Please provide a more detailed reflection (at least a sentence) to get an accurate engagement score.")
    else:
        with st.spinner("Analyzing your profile..."):
            try:
                result = predictor.predict(personality_data, reading_data)
                
                # Display Result
                st.success(f"### Recommended Career Path: **{result['final_recommendation']}**")
                
                # Detailed Scores
                st.subheader("Confidence Scores")
                
                # Create a DataFrame for the chart
                scores_df = pd.DataFrame({
                    'Domain': list(result['final_scores'].keys()),
                    'Score': list(result['final_scores'].values())
                })
                
                st.bar_chart(scores_df.set_index('Domain'))
                
                # Breakdown
                with st.expander("See Detailed Breakdown"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Personality Model Contribution**")
                        st.json(result['personality_scores'])
                    with c2:
                        st.write("**Reading Model Contribution**")
                        st.json(result['reading_scores'])
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Ensemble Unsupervised Learning Models")

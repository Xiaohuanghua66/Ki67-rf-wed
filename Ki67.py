import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##
# Load the trained model
model = joblib.load('RandomForestClassifier_best_model.pkl')  # 加载训练好的XGBoost模型
# Define feature options
location_options = {
    1: 'Right upper lobe (1)',
    2: 'Right middle lobe (2)',
    3: 'Right lower lobe (3)',
    4: 'Left upper lobe (4)',
    5: 'Left lower lobe (5)'
}

air_bronchogram_options = {
    0: 'NO (0)',
    1: 'Without tracheal deformation (1)',
    2: 'With tracheal deformation/traction (2)',
    3: 'Bronchiectasis/ectopic displacement (3)'
}

pleural_tags_options = {
    0: 'Type 0 (0)',
    1: 'Type Ⅰ (1)',
    2: 'Type Ⅱ (2)',
    3: 'Type Ⅲ (3)',
    4: 'Type Ⅳ (4)'
}

# Streamlit UI
st.title("CT-based Ki67 Predictor")

# Sidebar for input options
st.sidebar.header("CT Features Input")

# Input controls
age = st.sidebar.number_input("Age:", min_value=20, max_value=100, value=50)
smoking = st.sidebar.selectbox("Smoking History:", 
                             options=[0, 1, 2],
                             format_func=lambda x: ['Never (0)', 'Former (1)', 'Current (2)'][x])

surgical_history = st.sidebar.selectbox("Surgical History:",
                                      options=[0, 1, 2],
                                      format_func=lambda x: ['None (0)', 'Digestive tract cancer (1)', 'Other cancer (2)'][x])

emphysema = st.sidebar.selectbox("Emphysema/Pulmonary Blisters:",
                               options=[0, 1],
                               format_func=lambda x: ['No (0)', 'Yes (1)'][x])

tumor_indicators = st.sidebar.selectbox("Tumour Indicators:",
                                      options=[0, 1],
                                      format_func=lambda x: ['No (0)', 'Yes (1)'][x])

location = st.sidebar.selectbox("Location:", 
                              options=list(location_options.keys()),
                              format_func=lambda x: location_options[x])

lobulation = st.sidebar.selectbox("Lobulation:",
                                options=[0, 1],
                                format_func=lambda x: ['No (0)', 'Yes (1)'][x])

spiculation = st.sidebar.selectbox("Spiculation:",
                                 options=[0, 1, 2],
                                 format_func=lambda x: ['None (0)', 'Short spicules (1)', 'Long spicules (2)'][x])

airspace = st.sidebar.selectbox("Airspace:",
                              options=[0, 1],
                              format_func=lambda x: ['No (0)', 'Yes (1)'][x])

air_bronchogram = st.sidebar.selectbox("Air Bronchogram:",
                                     options=list(air_bronchogram_options.keys()),
                                     format_func=lambda x: air_bronchogram_options[x])

pleural_tags = st.sidebar.selectbox("Pleural Tags:",
                                  options=list(pleural_tags_options.keys()),
                                  format_func=lambda x: pleural_tags_options[x])

intraperi = st.sidebar.number_input("INTRAPERI_0-10mm:", 
                                  min_value=0.0, max_value=10.0, value=5.0)

resnet_roi = st.sidebar.number_input("Resnet101_ROI Score:",
                                   min_value=0.0, max_value=1.0, value=0.5)

# Process the input and make prediction
feature_values = [age, smoking, surgical_history, emphysema, tumor_indicators,
                 location, lobulation, spiculation, airspace, air_bronchogram,
                 pleural_tags, intraperi, resnet_roi]

features = np.array([feature_values])

if st.button("Predict Ki67 Status"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    # Display results
    st.write(f"**Predicted Ki67 Status:** {'High (Positive)' if predicted_class == 1 else 'Low (Negative)'}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate medical advice
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"The model predicts a high Ki67 proliferation index ({probability:.1f}%). "
            "This suggests potentially aggressive tumor behavior. "
            "Recommendation: Further histopathological evaluation and consideration of adjuvant therapy."
        )
    else:
        advice = (
            f"The model predicts a low Ki67 proliferation index ({probability:.1f}%). "
            "This suggests relatively indolent tumor behavior. "
            "Recommendation: Regular follow-up with imaging and clinical monitoring."
        )
    st.write(advice)
    
    # Visualization
    sample_prob = {
        'Low_Ki67': predicted_proba[0],
        'High_Ki67': predicted_proba[1]
    }
    
    plt.figure(figsize=(10, 3))
    bars = plt.barh(['Low Ki67', 'High Ki67'], 
                    [sample_prob['Low_Ki67'], sample_prob['High_Ki67']], 
                    color=['#2ecc71', '#e74c3c'])
    
    plt.title("Ki67 Proliferation Index Prediction", fontsize=20, fontweight='bold')
    plt.xlabel("Probability", fontsize=14, fontweight='bold')
    plt.ylabel("Categories", fontsize=14, fontweight='bold')
    
    for i, v in enumerate([sample_prob['Low_Ki67'], sample_prob['High_Ki67']]):
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, 
                 color='black', fontweight='bold')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    st.pyplot(plt)

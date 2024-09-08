import pickle
import numpy as np
import pandas as pd
import google.generativeai as gemini

gemini.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")


# Load the model and scaler from the pickle file
with open('RESPIRATORY_MODEL/Lung_Cancer/lung_cancer.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    ensemble_model = saved_data['model']
    scaler = saved_data['scaler']

# Load dataset to get feature names
lung_cancer = pd.read_csv("RESPIRATORY_MODEL/Lung_Cancer/cancer_patient_data.csv")
x = lung_cancer.drop(columns="Level", axis=1)

# Function to predict lung cancer and provide risk percentage and disease type based on input data
def predict_lung_cancer(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_df = pd.DataFrame([input_data_as_numpy_array], columns=x.columns)  # Create DataFrame with feature names
    input_data_scaled = scaler.transform(input_data_df)  # Scale input data
    
    # Predict using the ensemble model
    prediction = ensemble_model.predict(input_data_scaled)[0]
    
    # Get the prediction probabilities
    prediction_probabilities = ensemble_model.predict_proba(input_data_scaled)[0]
    
    # Determine lung cancer status and risk percentage
    if prediction == 0:
        risk = "The person is not having lung cancer"
        risk_percentage = 0  # If there is no cancer, set the risk percentage to 0
        disease_type = "No Lung Cancer"
    elif prediction == 1:
        risk = "The person is having lung cancer"
        risk_percentage = prediction_probabilities[1] * 100
        disease_type = "Lung Cancer"
    elif prediction == 2:
        risk = "You are having severe lung cancer, please visit the doctor."
        risk_percentage = prediction_probabilities[2] * 100
        disease_type = "Severe Lung Cancer"
    else:
        risk = "Prediction output is unrecognized."
        risk_percentage = -1  # An indication of error
        disease_type = "Unknown"
    
    return risk, risk_percentage, disease_type

# Get user input
age = int(input("Enter age: "))
gender = int(input("Enter gender (1 = Male, 0 = Female): "))
air_pollution = int(input("Enter air pollution level (1 to 4): "))
alcohol_use = int(input("Enter alcohol use level (1 to 4): "))
dust_allergy = int(input("Enter dust allergy level (1 to 4): "))
occupational_hazards = int(input("Enter occupational hazards level (1 to 4): "))
genetic_risk = int(input("Enter genetic risk level (1 to 4): "))
chronic_lung_disease = int(input("Enter chronic lung disease level (1 to 4): "))
balanced_diet = int(input("Enter balanced diet level (1 to 4): "))
obesity = int(input("Enter obesity level (1 to 4): "))
smoking = int(input("Enter smoking level (1 to 4): "))
passive_smoker = int(input("Enter passive smoker level (1 to 4): "))
chest_pain = int(input("Enter chest pain level (1 to 4): "))
coughing_of_blood = int(input("Enter coughing of blood level (1 to 4): "))
fatigue = int(input("Enter fatigue level (1 to 4): "))
weight_loss = int(input("Enter weight loss level (1 to 4): "))
shortness_of_breath = int(input("Enter shortness of breath level (1 to 4): "))
wheezing = int(input("Enter wheezing level (1 to 4): "))
swallowing_difficulty = int(input("Enter swallowing difficulty level (1 to 4): "))
clubbing_of_finger_nails = int(input("Enter clubbing of finger nails level (1 to 4): "))
frequent_cold = int(input("Enter frequent cold level (1 to 4): "))
dry_cough = int(input("Enter dry cough level (1 to 4): "))
snoring = int(input("Enter snoring level (1 to 4): "))

# Combine input values into a single list
input_data = [age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards, genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker, chest_pain, coughing_of_blood, fatigue, weight_loss, shortness_of_breath, wheezing, swallowing_difficulty, clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]

# Predict lung cancer status and cause
risk, risk_percentage, disease_type = predict_lung_cancer(input_data)

# Print the results
print("Risk:", risk)
print("Risk Percentage:", risk_percentage)
print("Problem:", disease_type)

# Function to generate prevention report
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        # Assume gemini is an API client that generates text
        response = gemini.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1000
        )
        
        report = response.result if hasattr(response, 'result') else None
        
        if not report:
            print("The response from the API did not contain a result.")
        
        return report
    except Exception as e:
        print(f"An error occurred: {e}")

# Generate the prevention report using the predicted risk, disease type, and age
prevention_report = generate_prevention_report(risk, disease_type, age)

if prevention_report:
    print("\nPrevention Report:")
    print(prevention_report)
else:
    print("No report was generated.")

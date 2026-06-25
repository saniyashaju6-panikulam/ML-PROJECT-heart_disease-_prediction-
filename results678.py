import joblib
import numpy as np

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = r"C:\Users\Ethnotech\Desktop\project_heart\Heart_disease_prediction\heart_result\heart_knn_model.pkl"
ENCODER_PATH = r"C:\Users\Ethnotech\Desktop\project_heart\Heart_disease_prediction\heart_result\label_encoders.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)

    print("✓ Model loaded successfully")
    print("✓ Encoders loaded successfully")

except Exception as e:
    print(f"Error: {e}")
    exit()

# ==========================================
# USER INPUT
# ==========================================

print("\nEnter Patient Details\n")

age = float(input("Age: "))
gender = input("Gender (Male/Female): ").strip()

chest_pain = input(
    "Chest Pain Type (Typical Angina/Atypical Angina/Non-Anginal Pain/Asymptomatic): "
).strip()

resting_bp = float(input("Resting BP: "))
cholesterol = float(input("Cholesterol: "))
fasting_bs = float(input("Fasting Blood Sugar: "))

resting_ecg = input(
    "Resting ECG (Normal/ST-T Wave Abnormality/Left Ventricular Hypertrophy): "
).strip()

max_hr = float(input("Max Heart Rate: "))

exercise_angina = input(
    "Exercise Angina (Yes/No): "
).strip()

oldpeak = float(input("Oldpeak: "))

slope = input(
    "Slope (Upsloping/Flat/Downsloping): "
).strip()

thal = input(
    "Thal (Normal/Fixed Defect/Reversible Defect): "
).strip()

patient_id = float(input("Patient ID: "))

# ==========================================
# ENCODE CATEGORICAL VALUES
# ==========================================

try:
    gender = encoders["gender"].transform([gender])[0]
    chest_pain = encoders["chest_pain_type"].transform([chest_pain])[0]
    resting_ecg = encoders["resting_ecg"].transform([resting_ecg])[0]
    exercise_angina = encoders["exercise_angina"].transform([exercise_angina])[0]
    slope = encoders["slope"].transform([slope])[0]
    thal = encoders["thal"].transform([thal])[0]

except Exception as e:
    print("\nEncoding Error!")
    print("Please enter values exactly as shown.")
    print(e)
    exit()

# ==========================================
# CREATE FEATURE ARRAY
# ==========================================

sample = np.array([
    age,
    gender,
    chest_pain,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    slope,
    thal,
    patient_id
]).reshape(1, -1)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(sample)

print("\n========== RESULT ==========")

if prediction[0] == 1:
    print("Prediction: HEART DISEASE DETECTED")
    print("\nRecommendation:")
    print("- Consult a cardiologist.")
    print("- Monitor blood pressure and cholesterol.")
    print("- Follow a healthy diet and exercise plan.")

else:
    print("Prediction: NO HEART DISEASE DETECTED")
    print("\nHealth Status:")
    print("- Low likelihood of heart disease.")
    print("- Continue maintaining a healthy lifestyle.")

if hasattr(model, "predict_proba"):
    confidence = max(model.predict_proba(sample)[0]) * 100
    print(f"\nModel Confidence: {confidence:.2f}%")

print("============================")
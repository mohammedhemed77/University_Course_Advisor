import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. & 2. Load and Aggregate (Same as your logic)
info = pd.read_csv('studentInfo.csv')
vle = pd.read_csv('studentVle.csv')
vle_grouped = vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()

# 3. Merge and Clean
df = pd.merge(info, vle_grouped, on=['id_student', 'code_module', 'code_presentation'], how='left').fillna(0)

# 4. Target Definition
df['target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)

# 5. Encoding (Saving Encoders to decode later for the advisor's speech)
encoders = {}
cols_to_encode = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 6. Model Training
X = df.drop(['id_student', 'final_result', 'target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)

# --- ENHANCEMENT: THE ADVISOR ENGINE ---

def ai_advisor_feedback(student_data, prediction_prob):
    """Generates human-like advice based on model features."""
    clicks = student_data['sum_click']
    prev_attempts = student_data['num_of_prev_attempts']
    
    if prediction_prob >= 0.75:
        status = "🌟 EXCELLENT STANDING"
        advice = "You are performing at a high level. Consider mentoring peers or looking into 'Distinction' level elective projects."
    elif 0.50 <= prediction_prob < 0.75:
        status = "✅ ON TRACK (CAUTION)"
        advice = "You are passing, but your engagement is inconsistent. Try to set a daily schedule for VLE activity to secure your grade."
    else:
        status = "⚠️ AT-RISK ALERT"
        # Logic-based personalized advice
        if clicks < 150:
            advice = "Critical: Your VLE engagement is significantly below average. Accessing course materials more frequently is the #1 way to improve."
        elif prev_attempts > 0:
            advice = "It looks like this module has been a challenge before. I recommend booking a session with a subject tutor immediately."
        else:
            advice = "You're struggling despite some activity. Let's review your study methods—consider the 'Active Recall' workshop."
            
    return status, advice

# --- TESTING WITH SAMPLES ---

print("\n" + "="*50)
print("UNIVERSITY AI ADVISOR - LIVE SAMPLES")
print("="*50)

# Take 3 diverse samples from the test set
samples = X_test.sample(3, random_state=10)
sample_probs = rf.predict_proba(samples)[:, 1]

for i in range(len(samples)):
    current_student = samples.iloc[i]
    prob = sample_probs[i]
    
    status, advice = ai_advisor_feedback(current_student, prob)
    
    print(f"\n[Student Profile {i+1}]")
    print(f"Engagement: {int(current_student['sum_click'])} clicks")
    print(f"Success Probability: {prob:.2%}")
    print(f"Advisor Status: {status}")
    print(f"Advisor Recommendation: {advice}")
    print("-" * 30)
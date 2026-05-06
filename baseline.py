import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load data
info = pd.read_csv('studentInfo.csv')
vle = pd.read_csv('studentVle.csv')

# 2. Aggregating VLE interactions (Total clicks per student)
# This is crucial because raw VLE data is too big for a simple RF
vle_grouped = vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()

# 3. Merge demographics with engagement data
df = pd.merge(info, vle_grouped, on=['id_student', 'code_module', 'code_presentation'], how='left').fillna(0)

# 4. Define target (Success vs. At-Risk)
# We treat 'Withdrawn' and 'Fail' as 0 (At-Risk)
df['target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)

# 5. Preprocessing: Encoding categorical features
le = LabelEncoder()
cols_to_encode = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col].astype(str))

# 6. Model Training
X = df.drop(['id_student', 'final_result', 'target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
#rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# Try increasing the number of trees and allowing them to grow deeper
rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)

# 7. Results
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 8. Feature Importance for the Advisor
importances = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
print(importances.sort_values(by='importance', ascending=False))
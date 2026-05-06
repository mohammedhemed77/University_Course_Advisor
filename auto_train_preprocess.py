import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the raw OULAD files
# Make sure these files are in your 'university student advisor' folder
print("Merging OULAD tables...")
info = pd.read_csv('studentInfo.csv')
vle = pd.read_csv('studentVle.csv')
assessments = pd.read_csv('assessments.csv')
student_assess = pd.read_csv('studentAssessment.csv')

# 2. Feature Engineering (Aggregation)
# Clicks: Sum total clicks per student per module
vle_feat = vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()

# Scores: Calculate weighted average score
assess_merged = pd.merge(student_assess, assessments, on='id_assessment')
assess_merged['weighted_score'] = (assess_merged['score'] * assess_merged['weight']) / 100
assess_feat = assess_merged.groupby(['id_student', 'code_module', 'code_presentation'])['weighted_score'].mean().reset_index()

# 3. Merge everything into one Master Table
df = pd.merge(info, vle_feat, on=['id_student', 'code_module', 'code_presentation'], how='left')
df = pd.merge(df, assess_feat, on=['id_student', 'code_module', 'code_presentation'], how='left')

# 4. Final Cleanup for AutoTrain
# AutoTrain likes the label to be named 'target'
df['target'] = df['final_result'].map({
    'Pass': 1, 
    'Distinction': 1, 
    'Fail': 0, 
    'Withdrawn': 0
})

# Fill missing values (important for Transformers)
df['sum_click'] = df['sum_click'].fillna(0)
df['weighted_score'] = df['weighted_score'].fillna(df['weighted_score'].mean())

# Drop the columns AutoTrain doesn't need
# Note: Keep the categorical strings! AutoTrain handles encoding for you.
cols_to_drop = ['id_student', 'final_result']
df_final = df.drop(columns=cols_to_drop)

# 5. Split and Save
train, test = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['target'])

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

print("✅ Success! 'train.csv' and 'test.csv' are ready for AutoTrain.")
print(f"Train size: {len(train)} | Test size: {len(test)}")
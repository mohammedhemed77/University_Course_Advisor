import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# ==========================================
# 1. DATA PREPROCESSING (The "x_train" part)
# ==========================================
print("Loading and preprocessing data...")

# Load core files
info = pd.read_csv('studentInfo.csv')
vle = pd.read_csv('studentVle.csv')

# Aggregate VLE interactions (Total clicks per student per module)
vle_grouped = vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()

# Merge demographics with engagement data
df = pd.merge(info, vle_grouped, on=['id_student', 'code_module', 'code_presentation'], how='left').fillna(0)

# Define Target: 1 for Success (Pass/Distinction), 0 for At-Risk (Fail/Withdrawn)
df['target'] = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)

# Encode Categorical Features
le = LabelEncoder()
categorical_cols = ['code_module', 'code_presentation', 'gender', 'region', 
                    'highest_education', 'imd_band', 'age_band', 'disability']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Drop identifiers and the original target string
X = df.drop(['id_student', 'final_result', 'target'], axis=1)
y = df['target']

# Split Data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Networks REQUIRE scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================
# 2. PYTORCH SETUP
# ==========================================

# Convert to Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# Define the Neural Network Architecture
class AdvisorNN(nn.Module):
    def __init__(self, input_dim):
        super(AdvisorNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Squashes output between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)

# Initialize Model, Loss, and Optimizer
model = AdvisorNN(X_train.shape[1])
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print("\nStarting Training...")
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {epoch_loss/len(train_loader):.4f}")

# ==========================================
# 4. EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t)
    y_pred = (y_pred_prob > 0.5).float()
    
    accuracy = (y_pred == y_test_t).sum() / len(y_test_t)
    print(f"\nFinal PyTorch Accuracy: {accuracy.item():.4f}")
    
    # Convert back to CPU/Numpy for the report
    print("\nClassification Report:")
    print(classification_report(y_test_t.numpy(), y_pred.numpy()))
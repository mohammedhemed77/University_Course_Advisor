# 🎓 University Student Advisor: Early Warning System (EWS)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-orange)](https://scikit-learn.org/)


## 📌 Project Overview
This project transforms raw academic data into an **AI Student Advisor**. By utilizing the **Open University Learning Analytics Dataset (OULAD)**, the system identifies "at-risk" students by analyzing behavioral patterns (clicks) and demographic data.

Unlike a standard predictor, this system acts as a **Digital Triage Tool**, providing specific recommendations to human advisors based on the model's findings.

## 🧠 How It Works
The "Brain" of this advisor is a **Random Forest Classifier** (and an optional **Transformer** via AutoTrain) that analyzes the relationship between engagement and success.

### Key Components:
*   **Behavioral Analysis (Clicks):** Uses `sum_click` from the VLE to measure student momentum.
*   **Demographic Context:** Includes region, education level, and socio-economic indicators (`imd_band`).
*   **Diagnostic Engine:** Calculates "Feature Importance" to tell the advisor *why* a student is struggling.
*   **Advisory Logic:** Translates 0/1 predictions into human-readable advice.

## 🚀 Features
- **Early Detection:** Identifies at-risk students before final assessments.
- **Explainable AI:** Uses feature importance to justify every alert.
- **Automated Triage:** Ranks students by "Probability of Success" so advisors can prioritize their workload.


## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-Learn, PyTorch, NumPy
- **Models:** Random Forest (Baseline)

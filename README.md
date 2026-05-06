# 🎓 University Student Advisor: Early Warning System (EWS)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-orange)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-AutoTrain-yellow)](https://huggingface.co/autotrain)

## 📌 Project Overview
This project transforms raw academic data into an **AI Student Advisor**. By utilizing the **Open University Learning Analytics Dataset (OULAD)**, the system identifies "at-risk" students by analyzing behavioral patterns (clicks) and demographic data.

Unlike a standard predictor, this system acts as a **Digital Triage Tool**, providing specific recommendations to human advisors based on the model's findings.

## 📊 Dataset: OULAD
The project utilizes the **Open University Learning Analytics Dataset (OULAD)**, one of the most comprehensive datasets in the field of Learning Analytics.

*   **Source:** [Kaggle - Student Demographics & Online Education Data](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad)
*   **Context:** Contains data from 22 modules, 20 presentations, and 32,593 students.
*   **Key Data Points Used:**
    *   **Demographics:** Gender, region, education level, age, and disability status.
    *   **Social Context:** Indices of Multiple Deprivation (IMD) to account for socio-economic factors.
    *   **Student Behavior (VLE):** Aggregated logs of student interactions (clicks) with the Virtual Learning Environment.
    *   **Performance:** Success/Failure labels used to train the Supervised Learning model.

## 🧠 How It Works
The "Brain" of this advisor is a **Random Forest Classifier** that analyzes the relationship between engagement and success.

### Key Components:
*   **Behavioral Analysis (Clicks):** Uses `sum_click` from the VLE to measure student momentum.
*   **Demographic Context:** Includes region, education level, and socio-economic indicators (`imd_band`).
*   **Diagnostic Engine:** Calculates "Feature Importance" to tell the advisor *why* a student is struggling.
*   **Advisory Logic:** Translates 0/1 predictions into human-readable advice and intervention strategies.

## 🚀 Features
- **Early Detection:** Identifies at-risk students before they reach final assessments.
- **Explainable AI:** Uses feature importance to justify every alert, ensuring advisors understand the "why" behind the risk.
- **Automated Triage:** Ranks students by "Probability of Success" so advisors can prioritize high-impact interventions.
- **Agentic Ready:** Architecture allows for easy integration with LLM-based advisors (e.g., Llama-3, GPT-4) for automated student outreach.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-Learn, PyTorch, NumPy
- **Models:** Random Forest (Baseline), Transformers (Experimental/AutoTrain)
- **Deployment:** Hugging Face Spaces & AutoTrain

## 🚦 Getting Started
1. **Prepare Data:** Place `studentInfo.csv` and `studentVle.csv` in the project root.
2. **Train Model:** Run the Python script to generate the model and view the Accuracy/Feature Importance.
3. **Simulate Advisor:** Use the sample prediction logic to see how the AI generates advice for hypothetical students.

---
**Bridging the gap between Data Science and Student Success.** 🚀

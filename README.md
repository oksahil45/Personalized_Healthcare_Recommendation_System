# 🩺 MEDISCOPE
### Personalized Healthcare & Medicine Recommendation System

<p align="center">

  <a href="https://mediscope-healthcare.streamlit.app" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>

  <a href="https://github.com/oksahil45/Personalized-Healthcare-Recommendation-System">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>

</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat-square&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white"/>

</p>

---

## 🌐 Live Application

### 🚀 [Open MEDISCOPE — Live Demo](https://mediscope-healthcare.streamlit.app)

> An interactive Streamlit-based healthcare intelligence prototype for patient assessment, disease-category prediction, healthcare analytics, and reference-based medicine information.

---

## 📌 Project Overview

**MEDISCOPE** is a Data Science and Machine Learning project that combines patient-level healthcare data, machine learning, reference-based medicine information, and interactive analytics into a single Streamlit application.

The system allows users to:

- 👤 Enter patient information
- 🧠 Run a machine learning-based disease-category prediction
- 📊 View prediction probabilities
- 💊 Explore reference-based medicine information
- 📈 Analyze healthcare data
- 🔍 Explore symptom and patient patterns
- 🤖 Review model configuration and performance
- 📋 Understand the dataset and project methodology

The application is designed as an **educational and portfolio demonstration of a healthcare ML workflow**, not as a clinical diagnostic or prescription system.

---

# ✨ Key Features

## 👤 1. Patient Assessment

Users can enter available patient attributes and symptoms through an interactive interface.

The system processes the information using the trained machine learning pipeline and generates a predicted disease category.

### Input examples

- Age
- Gender
- Fever
- Cough
- Fatigue
- Breathing difficulty
- Blood pressure category
- Cholesterol category

---

## 🧠 2. Disease Prediction

The application uses a **Random Forest Classifier** to generate disease-category predictions.

The prediction workflow includes:

```text
Patient Input
      ↓
Data Validation
      ↓
Preprocessing
      ↓
Categorical Encoding
      ↓
Missing-Value Handling
      ↓
Random Forest Classifier
      ↓
Disease Prediction
      ↓
Probability Ranking

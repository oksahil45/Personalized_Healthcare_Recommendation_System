# 🩺 MEDISCOPE
### Personalized Healthcare & Medicine Recommendation System

<p align="center">

  <a href="https://sahilsuman-mediscope-healthcare.streamlit.app">
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

### 🚀 [Open MEDISCOPE — Live Demo](https://sahilsuman-mediscope-healthcare.streamlit.app)

**MEDISCOPE** is an interactive healthcare intelligence application built using Python, Machine Learning and Streamlit.

The application provides a unified workspace for:

- 👤 Patient assessment
- 🧠 Disease-category prediction
- 💊 Medicine and care references
- 📊 Healthcare analytics
- 🤖 Machine learning model information
- 📈 Prediction probability analysis

> **Note:** This project is an educational and portfolio prototype. It is not intended for clinical diagnosis, medical treatment, or prescription decisions.

---

# 📌 Project Overview

MEDISCOPE is a Data Science and Machine Learning project designed to demonstrate how structured healthcare data can be transformed into an interactive decision-support application.

The project combines:

**Data Processing → Machine Learning → Prediction → Reference Information → Interactive Analytics**

into a single Streamlit application.

The goal is to demonstrate an end-to-end machine learning workflow rather than simply training a model in a notebook.

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Analyze structured healthcare data.
2. Perform data cleaning and preprocessing.
3. Prepare features for machine learning.
4. Build a disease-category classification model.
5. Generate prediction probabilities.
6. Create an interactive patient assessment interface.
7. Connect predictions with reference-based healthcare information.
8. Provide interactive healthcare analytics.
9. Deploy the application using Streamlit.
10. Demonstrate practical Data Science and Machine Learning skills.

---

# ✨ Key Features

## 👤 Patient Assessment

The Patient Assessment module allows users to enter available patient attributes and symptoms.

The application processes the submitted information through the trained machine learning pipeline.

### Example inputs

- Age
- Gender
- Fever
- Cough
- Fatigue
- Breathing difficulty
- Blood pressure category
- Cholesterol category
- Other available patient attributes

---

# 🧠 Disease Prediction

The application uses a machine learning classification pipeline to generate a predicted disease category.

### Prediction workflow

```text
Patient Information
        ↓
Input Validation
        ↓
Data Preprocessing
        ↓
Feature Encoding
        ↓
Missing-Value Handling
        ↓
Random Forest Classifier
        ↓
Disease Prediction
        ↓
Probability Ranking

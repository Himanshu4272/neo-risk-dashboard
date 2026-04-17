# 🚀 NASA Near-Earth Object (NEO) Risk Analysis Dashboard

This project is an end-to-end Data Science and Machine Learning application that analyzes Near-Earth Objects (asteroids) using real-world data from NASA.

It combines data collection, preprocessing, visualization, and machine learning with an interactive web dashboard built using Streamlit.

---

## 🌌 Project Overview

Near-Earth Objects (NEOs) are asteroids that come close to Earth's orbit and may pose potential threats. This project aims to analyze such objects and predict whether they are hazardous using data-driven techniques.

---

## 📊 Key Features

- 📥 Data collected using NASA API (13,000+ records)
- 🧹 Data cleaning and preprocessing
- ⚙️ Feature engineering (Risk Score, Size Category, Distance Category)
- 📈 Exploratory Data Analysis (EDA)
- 📊 15+ visualizations (scatter plots, histograms, heatmaps, etc.)
- 🤖 Machine Learning model (Random Forest Classifier)
- 🌐 Interactive web dashboard using Streamlit
- 🔮 Real-time asteroid risk prediction

---

## 🧠 Machine Learning

- Model: Random Forest Classifier
- Input Features:
  - Diameter
  - Velocity
  - Distance
- Output:
  - Hazardous (0 = Safe, 1 = Dangerous)

### 📊 Model Evaluation:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📊 Data Visualization

The project includes multiple types of visualizations:
- Scatter Plot (Diameter vs Velocity)
- Histogram (Distribution analysis)
- Bar Charts (Category comparison)
- Heatmap (Correlation analysis)
- Pie Chart (Hazardous distribution)

---

## 🌐 Streamlit Dashboard

An interactive dashboard is built using Streamlit which includes:
- KPI metrics (Total asteroids, hazardous count, risk score)
- Interactive charts
- Sidebar for real-time prediction
- Clean and responsive layout

---

## 🛠 Tech Stack

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="NASA NEO Dashboard", layout="wide")

st.title("🚀 NASA Near-Earth Object (NEO) Risk Analysis")
st.markdown("### 🌌 Explore asteroid risk patterns using Machine Learning")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("cleaned_neo_data.csv")

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['Risk_Score'] = df['diameter'] * df['velocity']

def size(x):
    if x < 50:
        return "Small"
    elif x < 200:
        return "Medium"
    else:
        return "Large"

df['Size_Category'] = df['diameter'].apply(size)

def dist(x):
    if x < 1000000:
        return "Very Close"
    elif x < 5000000:
        return "Close"
    else:
        return "Far"

df['Distance_Category'] = df['distance'].apply(dist)

# -------------------------------
# MACHINE LEARNING MODEL
# -------------------------------
X = df[['diameter','velocity','distance']]
y = df['hazardous']

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# SIDEBAR (PREDICTION)
# -------------------------------
st.sidebar.header("🔮 Predict Asteroid Risk")

d = st.sidebar.number_input("Diameter", float(df['diameter'].min()), float(df['diameter'].max()))
v = st.sidebar.number_input("Velocity", float(df['velocity'].min()), float(df['velocity'].max()))
dis = st.sidebar.number_input("Distance", float(df['distance'].min()), float(df['distance'].max()))

if st.sidebar.button("Predict"):
    result = model.predict([[d, v, dis]])
    if result[0] == 1:
        st.sidebar.error("⚠️ Hazardous Asteroid")
    else:
        st.sidebar.success("✅ Safe Asteroid")

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Asteroids", len(df))
col2.metric("Hazardous Asteroids", int(df['hazardous'].sum()))
col3.metric("Avg Risk Score", round(df['Risk_Score'].mean(), 2))

# -------------------------------
# VISUALIZATIONS
# -------------------------------

# Scatter Plot (Full Width)
st.subheader("📈 Diameter vs Velocity")
fig, ax = plt.subplots()
sns.scatterplot(x='diameter', y='velocity', hue='hazardous', data=df, ax=ax)
st.pyplot(fig)

# -------------------------------
# SIDE BY SIDE CHARTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Size Category")
    fig, ax = plt.subplots()
    df['Size_Category'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Distance Category")
    fig, ax = plt.subplots()
    df['Distance_Category'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# -------------------------------
# HISTOGRAMS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Diameter Distribution")
    fig, ax = plt.subplots()
    df['diameter'].hist(ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Velocity Distribution")
    fig, ax = plt.subplots()
    df['velocity'].hist(ax=ax)
    st.pyplot(fig)

# -------------------------------
# HEATMAP + PIE
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[['diameter','velocity','distance','Risk_Score']].corr(),
                annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("🥧 Hazardous Distribution")
    fig, ax = plt.subplots()
    df['hazardous'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# -------------------------------
# DATA TABLE
# -------------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(20))
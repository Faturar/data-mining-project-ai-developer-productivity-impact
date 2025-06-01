import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import numpy as np
import pickle
from scipy import stats

# Set page config
st.set_page_config(page_title="AI Developer Productivity Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("./ai_dev_productivity.csv")

df = load_data()

# Calculate productivity score
df["productivity_score"] = (0.5 * df["commits"] + 0.3 * df["task_success"] - 0.2 * df["bugs_reported"])

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", 
                                 "Regression Model", "Classification Model", 
                                 "Clustering Analysis", "AI Impact Analysis",
                                 "Coding Efficiency", "Distraction Analysis"])

# Page 1: Data Overview
if page == "Data Overview":
    st.title("AI Developer Productivity Analysis")
    st.subheader("Data Overview")
    
    st.write("### First 5 Rows of Data")
    st.write(df.head())
    
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    st.write("### Data Description")
    st.write(df.describe())

# Page 2: Exploratory Analysis
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    
    st.write("### Pairplot of Features vs Task Success")
    X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
    fig = sns.pairplot(df, y_vars=["task_success"], x_vars=X.columns)
    st.pyplot(fig)
    
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Page 3: Regression Model
elif page == "Regression Model":
    st.title("Productivity Score Prediction (Regression)")
    
    X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
    y = df["productivity_score"]
    
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.write(f"### R² Score: {model.score(X_test, y_test):.2f}")
    st.write("### Coefficients:")
    st.write(dict(zip(X.columns, model.coef_)))
    
    # Save model
    if st.button("Save Model"):
        filename = "model_productivity.pkl"
        pickle.dump(model, open(filename, "wb"))
        st.success(f"Model saved as {filename}")

# Page 4: Classification Model
elif page == "Classification Model":
    st.title("High Productivity Classification")
    
    df["high_productivity"] = (df["productivity_score"] > df["productivity_score"].median()).astype(int)
    
    X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, 
        df["high_productivity"],
        test_size=0.2, 
        random_state=42
    )
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    st.write("### Feature Importances for Predicting High Productivity")
    st.write(dict(zip(X.columns, clf.feature_importances_)))

# Page 5: Clustering Analysis
elif page == "Clustering Analysis":
    st.title("Developer Clustering Analysis")
    
    X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    st.write("### Cluster Profiles")
    st.write(df.groupby("cluster").mean())
    
    st.write("### Pairplot by Cluster")
    fig = sns.pairplot(df, hue="cluster", vars=X.columns)
    st.pyplot(fig)

# Page 6: AI Impact Analysis
elif page == "AI Impact Analysis":
    st.title("Impact of AI Usage on Developer Productivity")
    
    median_ai = df["ai_usage_hours"].median()
    high_ai = df[df["ai_usage_hours"] > median_ai]["productivity_score"]
    low_ai = df[df["ai_usage_hours"] <= median_ai]["productivity_score"]
    
    t_stat, p_value = ttest_ind(high_ai, low_ai)
    
    # Cohen's d function
    def cohens_d(a, b):
        n1, n2 = len(a), len(b)
        pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1 + n2 - 2))
        return (np.mean(a) - np.mean(b)) / pooled_std
    
    d = cohens_d(high_ai, low_ai) 
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x=df["ai_usage_hours"] > median_ai,
        y=df["productivity_score"],
        hue=df["ai_usage_hours"] > median_ai,
        palette="pastel",
        width=0.5,
        legend=False
    )
    
    plt.xticks(
        [False, True],
        [f"Low AI Usage\n(≤{median_ai:.1f} hours)", 
         f"High AI Usage\n(>{median_ai:.1f} hours)"]
    )
    plt.xlabel("AI Usage Group", fontsize=12)
    plt.ylabel("Productivity Score", fontsize=12)
    plt.title("Impact of AI Usage on Developer Productivity", fontsize=14)
    
    if p_value < 0.05:
        effect_size = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        plt.text(0.5, 0.95, 
                 f"p-value = {p_value:.3f}*\nCohen's d = {d:.2f} ({effect_size} effect)",
                 ha='center', va='top', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.text(0.5, 0.95, 
                 f"p-value = {p_value:.3f} (Not Significant)\nCohen's d = {d:.2f}",
                 ha='center', va='top', transform=plt.gca().transAxes)
    
    st.pyplot(plt)
    
    # Numerical results
    st.write("### Analysis Results:")
    st.write(f"- Average productivity (High AI group): {high_ai.mean():.2f}")
    st.write(f"- Average productivity (Low AI group): {low_ai.mean():.2f}")
    st.write(f"- Difference: {high_ai.mean() - low_ai.mean():.2f} points")
    st.write(f"- Effect size (Cohen's d): {d:.2f}")
    st.write(f"- Statistical significance: {'Significant' if p_value < 0.05 else 'Not significant'}")

# Page 7: Coding Efficiency
elif page == "Coding Efficiency":
    st.title("Coding Efficiency by AI Usage")
    
    # Data Preparation
    df['ai_usage_group'] = pd.cut(df['ai_usage_hours'], 
                                bins=[0,3,6,24],
                                labels=['Low (0-3h)', 'Optimal (3-6h)', 'High (>6h)'])
    
    df['coding_efficiency'] = df['commits'] / df['hours_coding']
    df['coding_efficiency'] = df['coding_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Remove groups with insufficient data
    group_counts = df['ai_usage_group'].value_counts()
    valid_groups = group_counts[group_counts >= 1].index
    df_filtered = df[df['ai_usage_group'].isin(valid_groups)]
    
    # Visualization
    st.write("### Coding Efficiency by AI Usage Group")
    plt.figure(figsize=(12, 6))
    
    box = sns.boxplot(
        data=df_filtered,
        x="ai_usage_group",
        y="coding_efficiency",
        hue="ai_usage_group", 
        palette="viridis",
        width=0.4,
        order=valid_groups,
        dodge=False, 
        legend=False 
    )
    
    swarm = sns.swarmplot(
        data=df_filtered,
        x="ai_usage_group",
        y="coding_efficiency",
        color="black",
        alpha=0.5,
        size=5,
        order=valid_groups
    )
    
    plt.title("Coding Efficiency Distribution by AI Usage Group\n(Box: Q1-Q3, Line: Median, Points: Individual Data)", pad=20)
    plt.xlabel("AI Usage Group")
    plt.ylabel("Commits per Hour")
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(plt)
    
    # Descriptive Statistics
    st.write("### Descriptive Statistics:")
    for group in valid_groups:
        group_data = df_filtered[df_filtered['ai_usage_group'] == group]['coding_efficiency']
        st.write(f"\n**Group: {group}**")
        st.write(f"- Observations (N): {len(group_data)}")
        st.write(f"- Mean: {group_data.mean():.2f}")
        st.write(f"- Standard deviation: {group_data.std():.2f}")
        st.write(f"- Minimum: {group_data.min():.2f}")
        st.write(f"- Median: {group_data.median():.2f}")
        st.write(f"- Maximum: {group_data.max():.2f}")

# Page 8: Distraction Analysis
elif page == "Distraction Analysis":
    st.title("Coding Efficiency: AI Usage vs Distraction Levels")
    
    # Data preparation
    df["ai_usage_hours"] = pd.to_numeric(df["ai_usage_hours"], errors='coerce')
    df["distractions"] = pd.to_numeric(df["distractions"], errors='coerce')
    df["coding_efficiency"] = pd.to_numeric(df["coding_efficiency"], errors='coerce')
    
    df = df.dropna(subset=["ai_usage_hours", "distractions", "coding_efficiency"])
    
    median_ai = df["ai_usage_hours"].median()
    df["ai_usage_group"] = np.where(
        df["ai_usage_hours"] > median_ai, 
        "High AI", 
        "Low AI"
    )
    
    df["distraction_level"] = pd.cut(df["distractions"], 
                                    bins=[0, 2, 4, 6], 
                                    labels=["Low (0-2h)", "Medium (2-4h)", "High (4-6h)"])
    
    # Plot
    st.write("### Coding Efficiency by Distraction Level and AI Usage")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df,
        x="distraction_level",
        y="coding_efficiency",
        hue="ai_usage_group",
        palette="pastel",
        order=["Low (0-2h)", "Medium (2-4h)", "High (4-6h)"]
    )
    
    plt.title("Coding Efficiency vs Distraction Level\n(Low vs High AI Usage)", pad=20)
    plt.xlabel("\nDistraction Level", fontsize=12)
    plt.ylabel("Commits per Hour\n", fontsize=12)
    plt.legend(title="AI Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine()
    st.pyplot(plt)
    
    # Descriptive Statistics
    st.write("### Detailed Statistics by Group")
    
    for level in ["Low (0-2h)", "Medium (2-4h)", "High (4-6h)"]:
        high_ai = df[(df["distraction_level"] == level) & (df["ai_usage_group"] == "High AI")]["coding_efficiency"]
        low_ai = df[(df["distraction_level"] == level) & (df["ai_usage_group"] == "Low AI")]["coding_efficiency"]
        
        if len(high_ai) > 1 and len(low_ai) > 1:  # Need at least 2 samples for t-test
            n_high = len(high_ai)
            n_low = len(low_ai)
            mean_high = high_ai.mean()
            mean_low = low_ai.mean()
            std_high = high_ai.std()
            std_low = low_ai.std()
            
            t_stat, p_val = stats.ttest_ind(high_ai, low_ai, equal_var=False)
            
            pooled_std = np.sqrt(((n_high-1)*std_high**2 + (n_low-1)*std_low**2) / (n_high + n_low - 2))
            d = (mean_high - mean_low) / pooled_std
            
            st.write(f"**DISTRACTION LEVEL: {level}**")
            st.write("-"*50)
            st.write(f"- High AI (N={n_high}): {mean_high:.2f} ± {std_high:.2f} commits/hour")
            st.write(f"- Low AI (N={n_low}): {mean_low:.2f} ± {std_low:.2f} commits/hour")
            st.write(f"- Difference: {mean_high - mean_low:.2f} commits/hour")
            st.write(f"- t-test: t = {t_stat:.2f}, p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
            st.write(f"- Effect size (Cohen's d): {d:.2f}")
            
            if abs(d) < 0.2:
                size_interp = "very small"
            elif abs(d) < 0.5:
                size_interp = "small"
            elif abs(d) < 0.8:
                size_interp = "medium"
            else:
                size_interp = "large"
            
            st.write(f"- Interpretation: {size_interp} effect size")
            st.write("\n" + "="*50 + "\n")
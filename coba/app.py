import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Set page config
st.set_page_config(page_title="AI Productivity Analysis", layout="centered")

# Title
st.title("📈 AI Usage vs Productivity Analysis")

# Your data - replace this with your actual data
# Example data structure:
# high_ai = productivity scores for high AI usage group
# low_ai = productivity scores for low AI usage group
high_ai = np.array([85, 88, 92, 95, 90, 87, 93, 91, 89, 94])  # Replace with your data
low_ai = np.array([78, 82, 75, 80, 77, 79, 81, 76, 74, 83])   # Replace with your data
median_ai = 5.0  # Replace with your actual median cutoff value

# Calculate Cohen's d
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std

d = cohens_d(high_ai, low_ai)
t_stat, p_value = ttest_ind(high_ai, low_ai)

# Display results
st.subheader("Statistical Results")

col1, col2 = st.columns(2)
with col1:
    st.metric("High AI Group Mean", f"{np.mean(high_ai):.2f}")
    st.metric("Low AI Group Mean", f"{np.mean(low_ai):.2f}")
with col2:
    st.metric("Difference", f"{np.mean(high_ai) - np.mean(low_ai):.2f}")
    st.metric("Effect Size (Cohen's d)", f"{d:.2f}")

st.metric("Statistical Significance", 
          "Significant" if p_value < 0.05 else "Not Significant",
          delta=f"p-value: {p_value:.4f}")

# Visualization
st.subheader("Productivity Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    x=["Low AI", "High AI"],
    y=[low_ai, high_ai],
    palette="pastel",
    width=0.5,
    ax=ax
)
ax.set_xlabel("AI Usage Group", fontsize=12)
ax.set_ylabel("Productivity Score", fontsize=12)
ax.set_title(f"Productivity by AI Usage (Cutoff: {median_ai} hours)", fontsize=14)

# Add annotation
if p_value < 0.05:
    effect_size = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    ax.text(0.5, 0.95, 
            f"p-value = {p_value:.3f}*\nCohen's d = {d:.2f} ({effect_size} effect)",
            ha='center', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
else:
    ax.text(0.5, 0.95, 
            f"p-value = {p_value:.3f} (Not Significant)\nCohen's d = {d:.2f}",
            ha='center', va='top', transform=ax.transAxes)

st.pyplot(fig)

# Data table
st.subheader("Raw Data Preview")
df = pd.DataFrame({
    "High AI Group": high_ai,
    "Low AI Group": low_ai
})
st.dataframe(df.T)  # Transposed for better display

# Interpretation guide
st.sidebar.markdown("""
### Interpretation Guide:
- **p-value < 0.05**: Significant difference
- **Cohen's d Effect Size**:
  - 0.2: Small effect
  - 0.5: Medium effect  
  - 0.8: Large effect
""")
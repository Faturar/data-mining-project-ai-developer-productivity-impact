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

df = pd.read_csv("./ai_dev_productivity.csv")


print(df.head())

df["productivity_score"] = (0.5 * df["commits"] + 0.3 * df["task_success"] - 0.2 * df["bugs_reported"])

print("\nMissing Values:\n", df.isnull().sum())

X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
y = df["productivity_score"]  # or "commits"

sns.pairplot(df, y_vars=["task_success"], x_vars=X.columns)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Variabel")
plt.show()

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.2f}")
print("Coefficients:", dict(zip(X.columns, model.coef_)))


df["high_productivity"] = (df["productivity_score"] > df["productivity_score"].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    df["high_productivity"],
    test_size=0.2, 
    random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Feature Importances:", dict(zip(X.columns, clf.feature_importances_)))


kmeans = KMeans(n_clusters=3)
df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_profile = df.groupby("cluster").mean()
print(cluster_profile)

sns.pairplot(df, hue="cluster", vars=X.columns)
plt.show()

median_ai = df["ai_usage_hours"].median()

# 1. Kelompok dengan penggunaan AI di atas median
high_ai = df[df["ai_usage_hours"] > median_ai]["productivity_score"]

# 2. Kelompok dengan penggunaan AI di bawah atau sama dengan median 
low_ai = df[df["ai_usage_hours"] <= median_ai]["productivity_score"]

# Melakukan independent t-test untuk membandingkan produktivitas kedua kelompok
t_stat, p_value = ttest_ind(high_ai, low_ai)

print(f"T-test p-value: {p_value:.3f}")

filename = "model_productivity.csv"
pickle.dump(model, open(filename, "wb"))

# Hitung Cohen's d
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std

d = cohens_d(high_ai, low_ai) 

# Uji t-test
t_stat, p_value = ttest_ind(high_ai, low_ai)

# Visualisasi
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
    [f"Penggunaan AI Rendah\n(≤{median_ai:.1f} jam)", 
     f"Penggunaan AI Tinggi\n(>{median_ai:.1f} jam)"]
)
plt.xlabel("Kelompok Penggunaan AI", fontsize=12)
plt.ylabel("Skor Produktivitas", fontsize=12)
plt.title("Dampak Penggunaan AI pada Produktivitas Developer", fontsize=14)

# Tambahkan anotasi statistik
if p_value < 0.05:
    effect_size = "kecil" if abs(d) < 0.5 else "sedang" if abs(d) < 0.8 else "besar"
    plt.text(0.5, 0.95, 
             f"p-value = {p_value:.3f}*\nCohen's d = {d:.2f} (efek {effect_size})",
             ha='center', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
else:
    plt.text(0.5, 0.95, 
             f"p-value = {p_value:.3f} (Tidak Signifikan)\nCohen's d = {d:.2f}",
             ha='center', va='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Print hasil numerik
print(f"Hasil Analisis:\n"
      f"- Rata-rata produktivitas kelompok High AI: {high_ai.mean():.2f}\n"
      f"- Rata-rata produktivitas kelompok Low AI: {low_ai.mean():.2f}\n"
      f"- Perbedaan: {high_ai.mean() - low_ai.mean():.2f} poin\n"
      f"- Effect size (Cohen's d): {d:.2f}\n"
      f"- Signifikansi statistik: {'Signifikan' if p_value < 0.05 else 'Tidak signifikan'}")



      # Data Preparation
df['ai_usage_group'] = pd.cut(df['ai_usage_hours'], 
                            bins=[0,3,6,24],
                            labels=['Low (0-3h)', 'Optimal (3-6h)', 'High (>6h)'])

df['coding_efficiency'] = df['commits'] / df['hours_coding']

# Handle potential division by zero
df['coding_efficiency'] = df['coding_efficiency'].replace([np.inf, -np.inf], np.nan)

# Remove groups with insufficient data (less than 2 observations)
group_counts = df['ai_usage_group'].value_counts()
valid_groups = group_counts[group_counts >= 1].index
df_filtered = df[df['ai_usage_group'].isin(valid_groups)]

# Visualisation
plt.figure(figsize=(12, 6))

# Create boxplot with proper hue assignment
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

# Add swarmplot
swarm = sns.swarmplot(
    data=df_filtered,
    x="ai_usage_group",
    y="coding_efficiency",
    color="black",
    alpha=0.5,
    size=5,
    order=valid_groups
)

plt.title("Distribusi Efisiensi Coding per Kelompok Pemakaian AI\n(Box: Q1-Q3, Line: Median, Points: Individual Data)", pad=20)
plt.xlabel("Kelompok Pemakaian AI")
plt.ylabel("Commits per Jam")
plt.grid(axis='y', alpha=0.3)
plt.show()

# Descriptive Statistics
print("Statistik Deskriptif:")
print("---------------------")
for group in valid_groups:
    group_data = df_filtered[df_filtered['ai_usage_group'] == group]['coding_efficiency']
    print(f"\nGroup: {group}")
    print(f"Jumlah observasi (N): {len(group_data)}")
    print(f"Rata-rata: {group_data.mean():.2f}")
    print(f"Standar deviasi: {group_data.std():.2f}")
    print(f"Minimum: {group_data.min():.2f}")
    print(f"Median: {group_data.median():.2f}")
    print(f"Maksimum: {group_data.max():.2f}")


    # Pastikan tipe data numerik
df["ai_usage_hours"] = pd.to_numeric(df["ai_usage_hours"], errors='coerce')
df["distractions"] = pd.to_numeric(df["distractions"], errors='coerce')
df["coding_efficiency"] = pd.to_numeric(df["coding_efficiency"], errors='coerce')

# Hapus baris dengan nilai hilang
df = df.dropna(subset=["ai_usage_hours", "distractions", "coding_efficiency"])

# Hitung median AI usage
median_ai = df["ai_usage_hours"].median()

# Buat kolom kategori AI usage
df["ai_usage_group"] = np.where(
    df["ai_usage_hours"] > median_ai, 
    "High AI", 
    "Low AI"
)

# Buat kolom tingkat gangguan
df["distraction_level"] = pd.cut(df["distractions"], 
                                bins=[0, 2, 4, 6], 
                                labels=["Low (0-2 Jam)", "Medium (2-4 Jam)", "High (4-6 Jam)"])

# Plot
plt.figure(figsize=(12, 7))
sns.boxplot(
    data=df,
    x="distraction_level",
    y="coding_efficiency",
    hue="ai_usage_group",
    palette="pastel",
    order=["Low (0-2 Jam)", "Medium (2-4 Jam)", "High (4-6 Jam)"]
)

plt.title("Efisiensi Ngoding vs Tingkat Gangguan\n(Penggunaan AI Rendah vs Tinggi)", pad=20)
plt.xlabel("\nTingkat Gangguan", fontsize=12)
plt.ylabel("Commits per Jam\n", fontsize=12)
plt.legend(title="Kelompok AI", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
plt.tight_layout()
plt.show()

# Descriptive Statistics
print("Statistik Deskriptif:")
print("---------------------")
print("")

for level in ["Low (0-2 Jam)", "Medium (2-4 Jam)", "High (4-6 Jam)"]:
    high_ai = df[(df["distraction_level"] == level) & (df["ai_usage_group"] == "High AI")]["coding_efficiency"]
    low_ai = df[(df["distraction_level"] == level) & (df["ai_usage_group"] == "Low AI")]["coding_efficiency"]
    
    # Hitung statistik deskriptif
    n_high = len(high_ai)
    n_low = len(low_ai)
    mean_high = high_ai.mean()
    mean_low = low_ai.mean()
    std_high = high_ai.std()
    std_low = low_ai.std()
    
    # Uji t independen
    t_stat, p_val = stats.ttest_ind(high_ai, low_ai, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_high-1)*std_high**2 + (n_low-1)*std_low**2) / (n_high + n_low - 2))
    d = (mean_high - mean_low) / pooled_std
    
    print(f"TINGKAT GANGGUAN: {level}")
    print("-"*50)
    print(f"High AI (N={n_high}): {mean_high:.2f} ± {std_high:.2f} commits/jam")
    print(f"Low AI (N={n_low}): {mean_low:.2f} ± {std_low:.2f} commits/jam")
    print(f"Perbedaan: {mean_high - mean_low:.2f} commits/jam")
    print(f"Uji-t: t = {t_stat:.2f}, p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
    print(f"Effect size (Cohen's d): {d:.2f}")
    
    # Interpretasi effect size
    if abs(d) < 0.2:
        size_interp = "sangat kecil"
    elif abs(d) < 0.5:
        size_interp = "kecil"
    elif abs(d) < 0.8:
        size_interp = "sedang"
    else:
        size_interp = "besar"
    
    print(f"Interpretasi: Perbedaan {size_interp}")
    print("\n" + "="*50 + "\n")
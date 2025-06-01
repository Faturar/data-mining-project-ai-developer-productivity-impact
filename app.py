import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Memuat model yang telah dilatih
@st.cache_resource
def load_model():
    with open('model_classifier_productivity.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title('Prediktor Produktivitas Developer')
st.subheader('Prediksi bagaimana penggunaan AI mempengaruhi produktivitas Anda')

# Membuat kolom input
col1, col2 = st.columns(2)

with col1:
    hours_coding = st.number_input('Jam yang dihabiskan untuk coding', min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    coffee_intake = st.number_input('Asupan kopi (mg)', min_value=0, max_value=1000, value=200, step=50)
    distractions = st.number_input('Tingkat gangguan (skala 1-10)', min_value=1, max_value=10, value=3)
    
with col2:
    sleep_hours = st.number_input('Jam tidur (malam sebelumnya)', min_value=0.0, max_value=12.0, value=7.0, step=0.5)
    bugs_reported = st.number_input('Bug yang dilaporkan hari ini', min_value=0, value=2)
    cognitive_load = st.slider('Beban kognitif (skala 1-10)', min_value=1, max_value=10, value=5)

ai_usage = st.slider('Penggunaan AI hari ini (jam)', min_value=0.0, max_value=10.0, value=2.0, step=0.5)

# Tombol prediksi
if st.button('Prediksi Produktivitas Saya'):
    # Membuat dataframe input
    input_data = pd.DataFrame({
        'ai_usage_hours': [ai_usage],
        'coffee_intake_mg': [coffee_intake],
        'distractions': [distractions],
        'sleep_hours': [sleep_hours],
        'bugs_reported': [bugs_reported],
        'cognitive_load': [cognitive_load]
    })
    
    # Membuat prediksi
    productivity = model.predict(input_data)[0]
    predicted_commits = productivity * hours_coding
    
    # Klasifikasi kelompok pengguna AI
    if ai_usage == 0:
        ai_group = "Non-Pengguna AI"
        recommendation = "Pertimbangkan untuk mencoba tools AI selama 1-2 jam untuk meningkatkan produktivitas"
    elif ai_usage < 2:
        ai_group = "Pengguna AI Ringan"
        recommendation = "Anda menggunakan AI dengan efektif. Coba tingkatkan penggunaan sedikit untuk tugas-tugas kompleks"
    elif ai_usage < 5:
        ai_group = "Pengguna AI Sedang"
        recommendation = "Rentang optimal. Pertahankan keseimbangan antara AI dan coding manual"
    else:
        ai_group = "Pengguna AI Berat"
        recommendation = "Pantau produktivitas Anda - terlalu banyak AI mungkin mengurangi peluang belajar mendalam"
    
    # Menampilkan hasil
    st.success(f"Prediksi produktivitas: {productivity:.2f} commit/jam")
    st.success(f"Perkiraan commit hari ini: {predicted_commits:.1f}")
    
    # Menampilkan klasifikasi pengguna AI
    st.subheader("Klasifikasi Pengguna AI")
    st.info(f"**Kelompok**: {ai_group}")
    st.info(f"**Rekomendasi**: {recommendation}")
    
    # Visualisasi pentingnya fitur
    st.subheader("Faktor yang paling mempengaruhi produktivitas")
    features = ['ai_usage_hours', 'coffee_intake_mg', 'distractions', 
                'sleep_hours', 'bugs_reported', 'cognitive_load']
    importances = model.feature_importances_
    
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title('Tingkat Kepentingan Fitur')
    ax.set_xlabel('Tingkat Kepentingan Relatif')
    st.pyplot(fig)
    
    # Menampilkan hubungan penggunaan AI dengan produktivitas
    st.subheader("Dampak Penggunaan AI")
    ai_hours = np.linspace(0, 10, 20)
    simulated_data = pd.DataFrame({
        'ai_usage_hours': ai_hours,
        'coffee_intake_mg': [coffee_intake]*20,
        'distractions': [distractions]*20,
        'sleep_hours': [sleep_hours]*20,
        'bugs_reported': [bugs_reported]*20,
        'cognitive_load': [cognitive_load]*20
    })
    simulated_prod = model.predict(simulated_data)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(ai_hours, simulated_prod)
    ax2.scatter(ai_usage, productivity, color='red', s=100)
    ax2.set_title('Produktivitas pada Tingkat Penggunaan AI Berbeda')
    ax2.set_xlabel('Jam Penggunaan AI')
    ax2.set_ylabel('Produktivitas (commit/jam)')
    ax2.grid(True)
    st.pyplot(fig2)

# Penjelasan tambahan
st.markdown("""
### Cara menggunakan tool ini:
1. Isi metrik harian Anda
2. Klik "Prediksi Produktivitas Saya"
3. Lihat bagaimana penggunaan AI mempengaruhi prediksi output Anda

**Produktivitas** diukur sebagai commit per jam coding.

**Kelompok Pengguna AI**:
- **Non-Pengguna AI**: 0 jam
- **Pengguna AI Ringan**: <2 jam  
- **Pengguna AI Sedang**: 2-5 jam
- **Pengguna AI Berat**: 5+ jam
""")
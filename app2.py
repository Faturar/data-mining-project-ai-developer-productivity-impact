import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Memuat model yang telah dilatih
@st.cache_resource
def load_model():
    try:
        # Productivity prediction model (Logistic Regression)
        with open('logistic_regression_productivity_model.pkl', 'rb') as file:
            model = pickle.load(file)
        # Scaler
        return model
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Function to determine optimal AI usage (updated for logistic regression)
def determine_optimal_usage(ai_usage, productivity_prob, hours_coding):
    if hours_coding == 0:
        return "No Coding"
    
    ai_ratio = ai_usage / hours_coding if hours_coding > 0 else 0
    
    if productivity_prob < 0.5:  # Low productivity
        if ai_ratio < 0.2:
            return "Underusing AI"
        elif ai_ratio > 0.5:
            return "Overusing AI"
        else:
            return "Optimal AI Usage"
    else:  # High productivity
        if ai_ratio < 0.1:
            return "Underusing AI"
        elif ai_ratio > 0.4:
            return "Overusing AI"
        else:
            return "Optimal AI Usage"

try:
    productivity_model, scaler = load_model()
except:
    st.error("Gagal memuat model. Pastikan file model tersedia.")
    st.stop()

st.title('Prediktor Produktivitas Developer')
st.subheader('Prediksi bagaimana penggunaan AI mempengaruhi produktivitas Anda')

# Membuat kolom input (maintaining original layout)
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
    try:
        # Create input dataframe (order must match training data)
        input_data = pd.DataFrame({
            'coffee_intake_mg': [coffee_intake],
            'hours_coding': [hours_coding],
            'distractions': [distractions],
            'sleep_hours': [sleep_hours],
            'bugs_reported': [bugs_reported],
            'cognitive_load': [cognitive_load],
            'ai_usage_hours': [ai_usage],
            'commits': [0]  # Placeholder, will be predicted
        })
        
        # Scale the input data
        input_scaled = scaler.transform(input_data.drop('commits', axis=1))
        
        # Make predictions
        productivity_prob = productivity_model.predict_proba(input_scaled)[0][1]  # Probability of being productive
        productivity_class = productivity_model.predict(input_scaled)[0]  # 0 or 1
        
        # Determine optimal AI usage
        usage_status = determine_optimal_usage(ai_usage, productivity_prob, hours_coding)
        
        recommendations = {
            "Underusing AI": "Anda bisa meningkatkan produktivitas dengan menggunakan lebih banyak AI (tambahkan 1-2 jam)",
            "Overusing AI": "Penggunaan AI yang berlebihan mungkin mengurangi pembelajaran mendalam (kurangi 1-2 jam)",
            "Optimal AI Usage": "Penggunaan AI Anda sudah optimal untuk produktivitas saat ini",
            "No Coding": "Tidak ada aktivitas coding hari ini"
        }
        
        recommendation = recommendations.get(usage_status, "")
        
        # Tampilkan hasil (maintaining original output format)
        st.success(f"Probabilitas produktivitas tinggi: {productivity_prob:.2f}")
        st.success(f"Klasifikasi: {'Produktif' if productivity_class == 1 else 'Tidak Produktif'}")
        
        # AI Usage analysis
        st.subheader("Analisis Penggunaan AI")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Level AI**: {ai_usage} jam/hari")
        with col_b:
            st.info(f"**Status**: {usage_status}")
        
        st.info(f"**Rekomendasi**: {recommendation}")
        
        # Visualisasi (maintaining original tabs)
        tab1, tab2 = st.tabs(["Pengaruh Faktor", "Dampak AI"])
        
        with tab1:
            # Feature importance untuk model produktivitas
            st.subheader("Faktor yang Mempengaruhi Produktivitas")
            
            try:
                coefficients = productivity_model.coef_[0]
                features = input_data.drop('commits', axis=1).columns
                
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': np.abs(coefficients)
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots()
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_title('Pengaruh Fitur Terhadap Produktivitas')
                ax.set_xlabel('Nilai Absolut Koefisien')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan feature importance: {e}")

        with tab2:
            # Dampak penggunaan AI
            st.subheader("Dampak Penggunaan AI Terhadap Produktivitas")
            ai_hours = np.linspace(0, 10, 20)
            
            # Create simulated data keeping other features constant
            simulated_data = pd.DataFrame({
                'coffee_intake_mg': [coffee_intake]*20,
                'hours_coding': [hours_coding]*20,
                'distractions': [distractions]*20,
                'sleep_hours': [sleep_hours]*20,
                'bugs_reported': [bugs_reported]*20,
                'cognitive_load': [cognitive_load]*20,
                'ai_usage_hours': ai_hours,
                'commits': [0]*20
            })
            
            try:
                # Scale and predict
                simulated_scaled = scaler.transform(simulated_data.drop('commits', axis=1))
                simulated_probs = productivity_model.predict_proba(simulated_scaled)[:, 1]
                
                fig2, ax2 = plt.subplots()
                ax2.plot(ai_hours, simulated_probs)
                ax2.scatter(ai_usage, productivity_prob, color='red', s=100)
                
                # Add optimal zone
                optimal_min = hours_coding * 0.2  # 20% of coding time
                optimal_max = hours_coding * 0.5  # 50% of coding time
                ax2.axvspan(optimal_min, optimal_max, color='green', alpha=0.1, label='Zona Optimal')
                
                ax2.set_title('Hubungan Penggunaan AI dan Probabilitas Produktivitas')
                ax2.set_xlabel('Jam Penggunaan AI')
                ax2.set_ylabel('Probabilitas Produktif')
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error saat membuat visualisasi dampak AI: {e}")
            
    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")

# Bagian informasi (maintaining original info section)
with st.expander("ℹ️ Tentang Aplikasi Ini"):
    st.markdown("""
    ### Cara Menggunakan Tool Ini:
    1. Isi semua metrik harian Anda
    2. Klik tombol **"Prediksi Produktivitas Saya"**
    3. Lihat hasil prediksi dan rekomendasi
    
    **Definisi Metrik:**
    - **Produktivitas**: Diprediksi sebagai probabilitas (0-1) menjadi produktif
    - **Beban Kognitif**: Tingkat kesulitan mental (1=sangat mudah, 10=sangat berat)
    - **Tingkat Gangguan**: Frekuensi gangguan saat bekerja (1=jarang, 10=sering)
    
    **Status Penggunaan AI:**
    - **Optimal**: Penggunaan AI seimbang dengan produktivitas
    - **Underusing**: Potensi peningkatan produktivitas dengan lebih banyak AI
    - **Overusing**: Terlalu bergantung pada AI, mungkin mengurangi pembelajaran
    
    **Catatan:** Model ini menggunakan Logistic Regression untuk memprediksi produktivitas.
    """)

# Catatan kaki
st.caption("Aplikasi ini menggunakan model machine learning untuk memprediksi produktivitas developer berdasarkan pola penggunaan AI.")
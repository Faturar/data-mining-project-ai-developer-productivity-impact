import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Memuat model yang telah dilatih
@st.cache_resource
def load_model():
    try:
        # Productivity prediction model
        with open('model_regressor_productivity.pkl', 'rb') as file:
            productivity_model = pickle.load(file)
        return productivity_model
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Function to determine optimal AI usage
def determine_optimal_usage(ai_usage, productivity, hours_coding):
    # Simple heuristic - can be replaced with a trained model
    if hours_coding == 0:
        return "No Coding"
    
    ai_ratio = ai_usage / hours_coding
    
    if productivity < 0.5:
        if ai_ratio < 0.2:
            return "Underusing AI"
        elif ai_ratio > 0.5:
            return "Overusing AI"
        else:
            return "Optimal AI Usage"
    else:
        if ai_ratio < 0.1:
            return "Underusing AI"
        elif ai_ratio > 0.4:
            return "Overusing AI"
        else:
            return "Optimal AI Usage"

try:
    productivity_model = load_model()
except:
    st.error("Gagal memuat model. Pastikan file model tersedia.")
    st.stop()

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
    try:
        # Create input dataframe
        productivity_input = pd.DataFrame({
            'ai_usage_hours': [ai_usage],
            'coffee_intake_mg': [coffee_intake],
            'distractions': [distractions],
            'sleep_hours': [sleep_hours],
            'bugs_reported': [bugs_reported],
            'cognitive_load': [cognitive_load]
        })
        
        # Make productivity prediction
        productivity = productivity_model.predict(productivity_input)[0]
        predicted_commits = productivity * hours_coding
        
        # Determine optimal AI usage
        usage_status = determine_optimal_usage(ai_usage, productivity, hours_coding)
        
        recommendations = {
            "Underusing AI": "Anda bisa meningkatkan produktivitas dengan menggunakan lebih banyak AI (tambahkan 1-2 jam)",
            "Overusing AI": "Penggunaan AI yang berlebihan mungkin mengurangi pembelajaran mendalam (kurangi 1-2 jam)",
            "Optimal AI Usage": "Penggunaan AI Anda sudah optimal untuk produktivitas saat ini",
            "No Coding": "Tidak ada aktivitas coding hari ini"
        }
        
        recommendation = recommendations.get(usage_status, "")
        
        # Tampilkan hasil
        st.success(f"Prediksi produktivitas: {productivity:.2f} commit/jam")
        st.success(f"Perkiraan commit hari ini: {predicted_commits:.1f}")
        
        # AI Usage analysis
        st.subheader("Analisis Penggunaan AI")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Level AI**: {ai_usage} jam/hari")
        with col_b:
            st.info(f"**Status**: {usage_status}")
        
        st.info(f"**Rekomendasi**: {recommendation}")
        
        # Visualisasi
        tab1, tab2 = st.tabs(["Pengaruh Faktor", "Dampak AI"])
        
        with tab1:
            # Feature importance untuk model produktivitas
            st.subheader("Faktor yang Mempengaruhi Produktivitas")
            
            try:
                # Get the actual features from the model
                features = productivity_model.feature_names_in_
                importances = productivity_model.feature_importances_
                
                fig, ax = plt.subplots()
                ax.barh(features, importances)
                ax.set_title('Tingkat Kepentingan Fitur (Produktivitas)')
                ax.set_xlabel('Pentingnya Relatif')
                st.pyplot(fig)
            except AttributeError as e:
                st.warning(f"Tidak dapat menampilkan feature importance: {e}")

        with tab2:
            # Dampak penggunaan AI
            st.subheader("Dampak Penggunaan AI Terhadap Produktivitas")
            ai_hours = np.linspace(0, 10, 20)
            
            # Create simulated data with the correct features
            simulated_data = pd.DataFrame({
                'ai_usage_hours': ai_hours,
                'coffee_intake_mg': [coffee_intake]*20,
                'distractions': [distractions]*20,
                'sleep_hours': [sleep_hours]*20,
                'bugs_reported': [bugs_reported]*20,
                'cognitive_load': [cognitive_load]*20,
            })
            
            try:
                simulated_prod = productivity_model.predict(simulated_data)
                
                fig2, ax2 = plt.subplots()
                ax2.plot(ai_hours, simulated_prod)
                ax2.scatter(ai_usage, productivity, color='red', s=100)
                
                # Add optimal zone
                optimal_min = hours_coding * 0.2  # 20% of coding time
                optimal_max = hours_coding * 0.5  # 50% of coding time
                ax2.axvspan(optimal_min, optimal_max, color='green', alpha=0.1, label='Zona Optimal')
                
                ax2.set_title('Hubungan Penggunaan AI dan Produktivitas')
                ax2.set_xlabel('Jam Penggunaan AI')
                ax2.set_ylabel('Produktivitas (commit/jam)')
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error saat membuat visualisasi dampak AI: {e}")
            
    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")

# Bagian informasi
with st.expander("ℹ️ Tentang Aplikasi Ini"):
    st.markdown("""
    ### Cara Menggunakan Tool Ini:
    1. Isi semua metrik harian Anda
    2. Klik tombol **"Prediksi Produktivitas Saya"**
    3. Lihat hasil prediksi dan rekomendasi
    
    **Definisi Metrik:**
    - **Produktivitas**: Diukur sebagai commit per jam coding
    - **Beban Kognitif**: Tingkat kesulitan mental (1=sangat mudah, 10=sangat berat)
    - **Tingkat Gangguan**: Frekuensi gangguan saat bekerja (1=jarang, 10=sering)
    
    **Status Penggunaan AI:**
    - **Optimal**: Penggunaan AI seimbang dengan produktivitas
    - **Underusing**: Potensi peningkatan produktivitas dengan lebih banyak AI
    - **Overusing**: Terlalu bergantung pada AI, mungkin mengurangi pembelajaran
    
    **Catatan:** Jam coding memengaruhi analisis optimalitas penggunaan AI.
    """)

# Catatan kaki
st.caption("Aplikasi ini menggunakan model machine learning untuk memprediksi produktivitas developer berdasarkan pola penggunaan AI.")
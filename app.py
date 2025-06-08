import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Memuat model dan scaler yang telah dilatih
@st.cache_resource
def load_model_and_scaler():
    try:
        # Productivity prediction model
        with open('logistic_regression_productivity_model.pkl', 'rb') as file:
            productivity_model = pickle.load(file)
        
        # Load the scaler (assuming you have one saved)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return productivity_model, scaler
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model/scaler: {e}")
        st.stop()

try:
    productivity_model, scaler = load_model_and_scaler()
except:
    st.error("Gagal memuat model/scaler. Pastikan file tersedia.")
    st.stop()

st.title('Prediktor Produktivitas Developer')
st.subheader('Prediksi bagaimana penggunaan AI mempengaruhi produktivitas Anda')

# Membuat kolom input
col1, col2 = st.columns(2)

with col1:
    hours_coding = st.number_input('Jam yang dihabiskan untuk coding', min_value=0.0, max_value=24.0, step=0.5, value=6.08)
    coffee_intake = st.number_input('Asupan kopi (mg)', min_value=0, max_value=1000, step=50, value=594)
    commit = st.number_input('Commit', min_value=0.0, step=0.5, value=3.0)
    ai_usage = st.slider('Penggunaan AI hari ini (jam)', min_value=0.0, max_value=10.0, step=0.5, value=0.91)
    
with col2:
    sleep_hours = st.number_input('Jam tidur (malam sebelumnya)', min_value=0.0, max_value=12.0, step=0.5, value=5.3)
    bugs_reported = st.number_input('Bug yang dilaporkan hari ini', min_value=0, value=0)
    distractions = st.number_input('Tingkat gangguan (skala 1-10)', min_value=1, max_value=10, value=1)
    cognitive_load = st.slider('Beban kognitif (skala 1-10)', min_value=1, max_value=10, value=7)

# Tombol prediksi
if st.button('Prediksi Produktivitas Saya'):
    try:
        # Create input dataframe with the correct column order
        productivity_input = pd.DataFrame([[hours_coding, coffee_intake, distractions, sleep_hours, 
                                          commit, bugs_reported, ai_usage, cognitive_load]],
                                        columns=['hours_coding', 'coffee_intake_mg', 'distractions', 
                                                'sleep_hours', 'commits', 'bugs_reported', 
                                                'ai_usage_hours', 'cognitive_load'])
        
        # Scale the input data
        productivity_input_scaled = scaler.transform(productivity_input)
        
        # Make predictions
        prediction = productivity_model.predict(productivity_input_scaled)
        probability = productivity_model.predict_proba(productivity_input_scaled)[:, 1]
        
        # Display results
        st.success(f"Prediksi: {'Produktif' if prediction[0] == 1 else 'Tidak Produktif'}")
        st.success(f"Probabilitas Produktif: {probability[0]:.2%}")
        
        # Visualisasi
        # tab1, tab2 = st.tabs(["Pengaruh Faktor", "Dampak AI"])
        
        # with tab1:
        #     # Feature coefficients untuk model logistic regression
        #     st.subheader("Koefisien Model (Pengaruh Faktor)")
            
        #     try:
        #         if hasattr(productivity_model, 'coef_'):
        #             feature_names = ['hours_coding', 'coffee_intake_mg', 'distractions', 
        #                            'sleep_hours', 'commits', 'bugs_reported', 
        #                            'ai_usage_hours', 'cognitive_load']
                    
        #             coefficients = productivity_model.coef_[0]
                    
        #             fig, ax = plt.subplots(figsize=(10, 6))
        #             ax.barh(feature_names, coefficients)
        #             ax.set_title('Koefisien Model Logistic Regression')
        #             ax.set_xlabel('Nilai Koefisien')
        #             ax.set_ylabel('Fitur')
        #             st.pyplot(fig)
        #         else:
        #             st.warning("Model tidak memiliki atribut koefisien")
        #     except Exception as e:
        #         st.warning(f"Tidak dapat menampilkan koefisien model: {e}")

        # with tab2:
        #     # Dampak penggunaan AI
        #     st.subheader("Dampak Penggunaan AI Terhadap Produktivitas")
        #     ai_hours = np.linspace(0, 10, 20)
            
        #     # Create simulated data with scaled values
        #     simulated_data = pd.DataFrame({
        #         'hours_coding': [hours_coding]*20,
        #         'coffee_intake_mg': [coffee_intake]*20,
        #         'distractions': [distractions]*20,
        #         'sleep_hours': [sleep_hours]*20,
        #         'commits': [commit]*20,
        #         'bugs_reported': [bugs_reported]*20,
        #         'ai_usage_hours': ai_hours,
        #         'cognitive_load': [cognitive_load]*20,
        #     })
            
            # try:
            #     # Scale and predict
            #     simulated_data_scaled = scaler.transform(simulated_data)
            #     simulated_prod = productivity_model.predict_proba(simulated_data_scaled)[:, 1]
                
            #     fig2, ax2 = plt.subplots()
            #     ax2.plot(ai_hours, simulated_prod)
            #     ax2.scatter(ai_usage, probability[0], color='red', s=100)
                
            #     # Add optimal zone
            #     optimal_min = hours_coding * 0.2
            #     optimal_max = hours_coding * 0.5
            #     ax2.axvspan(optimal_min, optimal_max, color='green', alpha=0.1, label='Zona Optimal')
                
            #     ax2.set_title('Hubungan Penggunaan AI dan Probabilitas Produktivitas')
            #     ax2.set_xlabel('Jam Penggunaan AI')
            #     ax2.set_ylabel('Probabilitas Produktivitas')
            #     ax2.grid(True)
            #     ax2.legend()
            #     st.pyplot(fig2)
            # except Exception as e:
            #     st.error(f"Error saat membuat visualisasi dampak AI: {e}")
            
    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")
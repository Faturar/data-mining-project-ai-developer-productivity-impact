import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_model_and_scaler():
    try:
        with open('logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model/scaler: {e}")
        st.stop()

try:
    model, scaler = load_model_and_scaler()
except:
    st.error("Gagal memuat model/scaler. Pastikan file tersedia.")
    st.stop()

st.title('Prediktor Produktivitas Developer')
st.subheader('Prediksi bagaimana AI mempengaruhi produktivitas')

col1, col2 = st.columns(2)

with col1:
    hours_coding = st.number_input('Jam yang dihabiskan untuk coding', min_value=0.0, max_value=24.0, step=0.5, value=6.0)
    coffee_intake = st.number_input('Asupan kopi (mg)', min_value=0, max_value=1000, step=25, value=500)
    commit = st.number_input('Commit', min_value=0.0, max_value=20.0,step=1.0, value=3.0)
    ai_usage = st.slider('Penggunaan AI hari ini (jam)', min_value=0.0, max_value=10.0, step=0.25, value=1.0)

with col2:
    sleep_hours = st.number_input('Jam tidur (malam sebelumnya)', min_value=0.0, max_value=12.0, step=0.25, value=7.0)
    bugs_reported = st.number_input('Bug yang dilaporkan hari ini', min_value=0, max_value=20, step=1, value=0)
    distractions = st.number_input('Tingkat gangguan (skala 1-10)', min_value=1, max_value=10, step=1, value=3)
    cognitive_load = st.slider('Beban kognitif (skala 1-10)', min_value=1, max_value=10, step=1, value=5)

if st.button('Prediksi Produktivitas Saya'):
    try:
        productivity_input = pd.DataFrame([[hours_coding, coffee_intake, distractions, sleep_hours, 
                                          commit, bugs_reported, ai_usage, cognitive_load]],
                                        columns=['hours_coding', 'coffee_intake_mg', 'distractions', 
                                                'sleep_hours', 'commits', 'bugs_reported', 
                                                'ai_usage_hours', 'cognitive_load'])
        
        productivity_input_scaled = scaler.transform(productivity_input)
        
        # Prediksi
        prediction = model.predict(productivity_input_scaled)
        probability = model.predict_proba(productivity_input_scaled)[:, 1]
        
        with col1:
            st.success(f"Prediksi: {'Produktif' if prediction[0] == 1 else 'Tidak Produktif'}")
            
        with col2:
            st.success(f"Probabilitas Produktif: {probability[0]:.2%}")
        
        # Visualisasi
        tab1, tab2 = st.tabs(["Pengaruh Faktor", "Dampak AI"])
        
        with tab1:
            st.subheader("Koefisien Model (Pengaruh Faktor)")
            
            try:
                if hasattr(model, 'coef_'):
                    feature_names = ['hours_coding', 'coffee_intake_mg', 'distractions', 
                                   'sleep_hours', 'commits', 'bugs_reported', 
                                   'ai_usage_hours', 'cognitive_load']
                    
                    coefficients = model.coef_[0]
            
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefficients,
                        'Absolute_Coeff': np.abs(coefficients)
                    }).sort_values('Absolute_Coeff', ascending=True)
                    
                    sorted_features = feature_importance['Feature'].values
                    sorted_coefficients = feature_importance['Coefficient'].values
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(sorted_features, sorted_coefficients)
                    
                    for i, (feature, coeff) in enumerate(zip(sorted_features, sorted_coefficients)):
                        color = 'green' if coeff > 0 else 'red'
                        ax.barh(feature, coeff, color=color)
                        
                        ax.text(coeff, i, f" {coeff:.3f}",
                            ha='left' if coeff > 0 else 'right',
                            va='center', color='black')
                    
                    ax.set_title('Koefisien Model Logistic Regression (Diurutkan)')
                    ax.set_xlabel('Nilai Koefisien')
                    ax.set_ylabel('Fitur')
                    ax.grid(axis='x', alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    with st.expander("📊 Detail Koefisien"):
                        st.dataframe(feature_importance.style.format({
                            'Coefficient': '{:.3f}',
                            'Absolute_Coeff': '{:.3f}'
                        }))
                else:
                    st.warning("Model tidak memiliki atribut koefisien")
            except Exception as e:
                st.warning(f"Tidak dapat menampilkan koefisien model: {e}")

        with tab2:
            st.subheader("Dampak Penggunaan AI Terhadap Produktivitas")
            ai_hours = np.linspace(0, min(10, hours_coding*1.5), 50) 
            
            simulated_data = pd.DataFrame({
                'hours_coding': [hours_coding]*len(ai_hours),
                'coffee_intake_mg': [coffee_intake]*len(ai_hours),
                'distractions': [distractions]*len(ai_hours),
                'sleep_hours': [sleep_hours]*len(ai_hours),
                'commits': [commit]*len(ai_hours),
                'bugs_reported': [bugs_reported]*len(ai_hours),
                'ai_usage_hours': ai_hours,
                'cognitive_load': [cognitive_load]*len(ai_hours),
            })
            
            try:
                simulated_data_scaled = scaler.transform(simulated_data)
                simulated_prod = model.predict_proba(simulated_data_scaled)[:, 1]
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                ax2.plot(ai_hours, simulated_prod, 
                        linewidth=2.5, 
                        color='#1f77b4',
                        label='Probabilitas Produktivitas')
                
                ax2.scatter(ai_usage, probability[0], 
                           color='red', 
                           s=150,
                           zorder=5,
                           label='Penggunaan Anda')
                
                optimal_min = max(0, hours_coding * 0.2)
                optimal_max = min(ai_hours[-1], hours_coding * 0.5)
                ax2.axvspan(optimal_min, optimal_max, 
                           color='green', 
                           alpha=0.15, 
                           label='Zona Optimal AI')
                
                ax2.axhline(y=0.5, 
                           color='orange', 
                           linestyle='--', 
                           alpha=0.7,
                           label='Batas Produktivitas')
                
                ax2.annotate(f'Optimal: {optimal_min:.1f}-{optimal_max:.1f} jam',
                            xy=((optimal_min+optimal_max)/2, 0.1),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
                
                ax2.set_title('Analisis Dampak Penggunaan AI pada Produktivitas', pad=20)
                ax2.set_xlabel('Jam Penggunaan AI (per hari)')
                ax2.set_ylabel('Probabilitas Produktif')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')
                ax2.set_ylim(0, 1)
                
                st.pyplot(fig2)
                
                with st.expander("🔍 Interpretasi Hasil"):
                    st.markdown(f"""
                    - **Zona Optimal** ({optimal_min:.1f}-{optimal_max:.1f} jam):  
                      Penggunaan AI dalam rentang ini memberikan keseimbangan terbaik antara bantuan AI dan keterlibatan manusia.
                    - **Di bawah optimal**: Potensi underutilization alat AI
                    - **Di atas optimal**: Risiko ketergantungan berlebihan pada AI
                    - Titik merah menunjukkan penggunaan aktual Anda ({ai_usage} jam)
                    """)
                    
            except Exception as e:
                st.error(f"Error saat membuat visualisasi dampak AI: {e}")
            
    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")
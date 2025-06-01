import pickle
import streamlit as st

model = pickle.load(open('model_productivity.csv', 'rb'))

st.title('Toyota Car Price Prediction')

mileage = st.number_input('Masukan jarak tempuh kendaraan (KM)')
engineSize = st.number_input('Masukan kapasitas mesin')

predict = ''

if st.button('Hitung Harga'):
    predict = model.predict([[engineSize, mileage]])
    st.write('Harga kendaraan dalam dollar: ', predict)
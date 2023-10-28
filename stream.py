import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
life_model = pickle.load(open('estimasi_angka_harapan_hidup.sav','rb'))
image = Image.open('banner.jpeg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Angka Harapan Hidup Suatu Negara')


schooling = st.number_input('rata2 pendidikan masyarakat(tahun) :')
 
icr  = st.number_input('Indeks pendapatan+akses sumber daya :')
 
gdp  = st.number_input('GDP Negara :')

diphtheria = st.number_input('Cakupan Imunisasi Diphteria :')

bmi = st.number_input('rata-rata indeks bmi masyarakat :')

pe = st.number_input('PDB Negara untuk kesahatan :')

Alcohol = st.number_input('rata-rata konsumsi Alcohol :')

Status = st.number_input('Status (negara berkembang=0/maju=1) :')

#code untuk estimasi
ins_est=''

#membuat button
if st.button('Estimasi Harapan Hidup'):
        life_pred = life_model.predict([[schooling,icr,gdp,diphtheria,bmi,pe,Alcohol,Status]])

        st.success(f'Estimasi Harapan Hidup : {life_pred[0]:.2f}')
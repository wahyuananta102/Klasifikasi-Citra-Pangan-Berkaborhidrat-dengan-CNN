import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st
from keras.models import load_model

#Model prediksi
pangan = ['banana', 'cassava', 'corn', 'potatoes', 'ubi']
model = load_model("Carbohydrate_Food_Recog_Model.keras")

def model_prediction(test_image):
    input_image = tf.keras.utils.load_img(test_image, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, axis=0)
    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])
    with open(os.path.join('New Data', test_image.name), 'wb') as f:
        f.write(test_image.getbuffer())
    outcome = 'Hasil dari prediksi gambar adalah ' + pangan[np.argmax(result)] + ' dengan akurasi ' + str(np.max(result))
    return outcome
#Sidebar
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#Main Page
if (app_mode=="Home"):
    st.header('Klasifikasi Pangan Berkarbohidrat dengan CNN Models')
    st.markdown("""
       <div style="width: 800px; overflow-x: auto;">
    Aplikasi ini dapat digunakan untuk mengklasifikasi kesegaran dari pangan yang mengandung karbohidrat
    </div>
        """, unsafe_allow_html=True)
    image_path = "home_image.jpg"
    st.image(image_path)
elif (app_mode=="About Project"):
    st.header("About Project")
    st.markdown("""
           <div style="width: 800px; overflow-x: auto;">
           Ketahanan pangan adalah suatu kondisi di mana semua orang memiliki akses yang aman dan berkelanjutan terhadap makanan pokok
           yang mengandung karbohidrat sebagai penggantinya nasi cukup untuk memenuhi kebutuhan gizi dan preferensi makanan mereka 
           untuk hidup sehat dan aktif. Pertanian memainkan peran penting dalam mencapai ketahanan pangan dengan menyediakan makanan 
           bagi populasi yang terus meningkat. Beberapa strategi utama dalam pertanian untuk ketahanan pangan: meningkatkan produksi, 
           menjaga kelestarian lingkungan,  memperkuat rantai pasokan, mendukung petani kecil, mengurangi kemiskinan. Pertanian untuk 
           ketahanan pangan adalah upaya global yang membutuhkan kerjasama dari berbagai pihak, termasuk pemerintah, organisasi 
           internasional, sektor swasta, dan masyarakat sipil. Klasifikasi gambar merupakan salah satu metode dalam kecerdasan 
           buatan (AI), di mana model AI dilatih untuk mengidentifikasi dan mengkategorikan objek dalam gambar. 
           Kemudian prosesnya melibatkan beberapa langkah seperti pengumpulan data,  pra-pemrosesan gambar, ekstraksi fitur, 
           pelatihan model, sampai membentuk kategori gambar melalui hasil klasifikasi. Klasifikasi gambar memiliki banyak aplikasi, 
           seperti: penyortiran gambar yang bersih, deteksi objek, dan pencocokan gambar objek.
               </div>
        """, unsafe_allow_html=True)
    st.subheader("About Dataset")
    st.markdown("""
       <div style="width: 800px; overflow-x: auto;">
       Data proyek ini didapatkan dari situs kaggle dan juga dari scrapping. Dari kaggle dan dari scrapping didapatkan total 500 data gambar.
           <br>Dataset dari proyek ini berisikan gambar dari pangan dibawah ini:
           </div>
    """, unsafe_allow_html=True)
    st.code("Banana --> Pisang")
    st.code("Corn --> Jagung")
    st.code("Potatoes --> Kentang")
    st.code("Cassava --> Singkong")
    st.code("Ubi")
    st.subheader("Modeling")
    st.text("Dataset dari proyek ini terdiri dari total 500 data yang dibagi menjadi:")
    st.text("1. 400 data digunakan untuk Training")
    st.text("2. 100 data digunakan untuk Validasi")
    st.text("Tahapan-Tahapan yang dilakukan untuk Modeling adalah:")
    st.text("1. Menentukan Image Size, Batch Size dan Pembagian Dataset ")
    st.text("2. Caching, Shuffling, dan Prefetching ")
    st.text("3. Augmentasi Data ")
    st.text("4. Pembangunan Model CNN")
    st.text("5. Pelatihan Model ")

elif (app_mode=="Prediction"):
    st.header("Prediction")
    test_image=st.file_uploader("Pilihlah Gambar:")
    if test_image is not None:
        st.image(test_image)
    if(st.button("Show Prediction")):
        st.write("Prediksi dari gambar")
        st.markdown(model_prediction(test_image))

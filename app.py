import streamlit as st
import pandas as pd
import joblib
import sys
import streamlit as st

st.write(sys.version)

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS CUSTOM
# =========================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background-color: #eaf6ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #5b9bd5;
    border-right: 2px solid #4178a9;
}

/* Sidebar tidak bisa di-hide */
[data-testid="collapsedControl"] {
    display: none;
}

/* Hilangkan footer bawaan */
footer {
    visibility: hidden;
}

/* Semua tulisan utama hitam */
html, body, p, label, div {
    color: black !important;
}

/* Judul */
h1, h2, h3 {
    color: black !important;
}

/* =========================
   SIDEBAR TEXT PUTIH
========================= */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* =========================
   SELECTBOX WARNA PUTIH
========================= */
.stSelectbox div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 8px;
    color: black !important;
}

.stSelectbox svg {
    fill: black !important;
}

/* Dropdown option */
ul[role="listbox"] {
    background-color: white !important;
    color: black !important;
}

/* Option saat dipilih */
li[role="option"] {
    background-color: white !important;
    color: black !important;
}

/* Hover option */
li[role="option"]:hover {
    background-color: #d6ecff !important;
    color: black !important;
}

/* =========================
   INPUT BOX
========================= */
.stTextInput input,
.stNumberInput input {
    color: black !important;
    background-color: white !important;
    border-radius: 8px;
}

/* =========================
   BUTTON
========================= */
.stButton > button {
    background-color: #4da6ff;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1f7ae0;
    color: white !important;
}

/* =========================
   DATAFRAME
========================= */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
}

            # =========================
# STYLE MENU DATAFRAME
# =========================

[data-testid="stDataFrame"] [role="menu"] {
    background-color: #1e1e1e !important;
    color: white !important;
}

[data-testid="stDataFrame"] [role="menu"] * {
    color: white !important;
}

[data-testid="stDataFrame"] [role="menuitem"] {
    color: white !important;
}

[data-testid="stDataFrame"] [role="menuitem"]:hover {
    background-color: #404040 !important;
}
           
/* =========================
   COPYRIGHT
========================= */
.copyright {
    position: fixed;
    bottom: 15px;
    left: 20px;
    font-size: 13px;
    color: white !important;
    z-index: 999999;
}

/* =========================
   INPUT DAN SELECTBOX PUTIH
========================= */

/* Number input */
.stNumberInput input {
    background-color: white !important;
    color: black !important;
}

/* Text input */
.stTextInput input {
    background-color: white !important;
    color: black !important;
}

/* Selectbox utama */
div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
    border-radius: 8px !important;
}

/* Isi selectbox */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* Dropdown popup */
ul[role="listbox"] {
    background-color: white !important;
}

/* Pilihan dropdown */
li[role="option"] {
    background-color: white !important;
    color: black !important;
}

/* Hover dropdown */
li[role="option"]:hover {
    background-color: #d6ecff !important;
    color: black !important;
}

/* Label input */
label {
    color: black !important;
}

/* Placeholder */
input::placeholder {
    color: gray !important;
}

/* Panah dropdown */
svg {
    fill: black !important;
}
            
/* =========================
   BUTTON +/- NUMBER INPUT
========================= */

/* Tombol + dan - */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #b0b0b0 !important;
}

/* Hover tombol */
button[data-testid="stNumberInputStepUp"]:hover,
button[data-testid="stNumberInputStepDown"]:hover {
    background-color: #d6ecff !important;
    color: black !important;
}

/* Icon + dan - */
button[data-testid="stNumberInputStepUp"] svg,
button[data-testid="stNumberInputStepDown"] svg {
    fill: black !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# COPYRIGHT
# =========================
st.sidebar.markdown("""
<div class="copyright">
© 2026 by Niken Amalia
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD FILE
# =========================
model = joblib.load("bestmodel_svm.pkl")
scaler = joblib.load("normminmax_scaler.pkl")
selected_features = joblib.load("hasil_ig.pkl")

# =========================
# LOAD DATASET
# =========================
data = pd.read_csv("HeartDisease.csv")
clean = pd.read_csv("dataset_clean.csv")
data_scaler= pd.read_csv("data_hasil_normalisasi_minmax.csv")

# =========================
# SIDEBAR
# =========================
# FOTO PROFIL
st.sidebar.image(
    "logo.jpeg",  # ganti dengan nama file gambar kamu
    width=150
)

# =========================
# SIDEBAR MENU
# =========================

st.sidebar.title("Menu")

if "menu" not in st.session_state:
    st.session_state.menu = "Deskripsi"

# Tombol menu
if st.sidebar.button("📄 Deskripsi", use_container_width=True):
    st.session_state.menu = "Deskripsi"

if st.sidebar.button("🔍 Prediksi", use_container_width=True):
    st.session_state.menu = "Prediksi"

menu = st.session_state.menu


# =========================
# MENU DESKRIPSI
# =========================
if menu == "Deskripsi":

    st.title("KLASIFIKASI PENYAKIT JANTUNG MENGGUNAKAN METODE SUPPORT VECTOR MACHINE (SVM) DENGAN SELEKSI FITUR INFORMATION GAIN")

    st.write("""
    Penyakit jantung merupakan salah satu penyebab kematian terbanyak didunia, yang disebabkan oleh sejumlah faktor.
    Klasifikasi adalah sebuah proses pengelompokkan data ke dalam suatu kelas. Dalam sebuah penelitian tentu tidak 
    selalu menghasilkan nilai akurasi terbaik maka dapat ditingkatkan dengan menggunakan seleksi fitur Information Gain 
    untuk memilih fitur yang relevan. Metode usulan yang digunakan untuk Klasifikasi yaitu SVM. Data yang digunakan berasal dari
    kaggle berjumlah 4.238 data. Selain itu, etidakseimbangan data akan ditangani menggunakan teknik oversampling yaitu SMOTE.

    """)

    st.header("Dataset")
    st.write("""
    Dataset yang akan digunakan pada penelitian ini adalah Heart Disease dari website
    Kaggle Repository ( https://www.kaggle.com/datasets/dileep070/heart-diseaseprediction-using-logistic-regression/data ). Dataset tersebut terdiri dari 4238 data
    dengan 15 fitur serta 1 fitur diagnosa.


    """)
    st.dataframe(data.head())

    st.header("Preprocessing")
    st.subheader("Penanganan Missing Value")
    st.write("""
    Dalam penanganan dataset yang digunakan dalam penelitian ini, terdapat
    sejumlah nilai yang tidak lengkap. Teknik yang digunakan untuk mengatasi
    masalah nilai yang hilang adalah imputasi, dimana metode mean dipilih karena
    merupakan salah satu pendekatan yang paling umum dan sederhana mangganti nilai
    yang hilang


    """)
    st.dataframe(data.iloc[[14]])
    st.dataframe(clean.iloc[[14]])

    st.subheader("Normalisasi Min-Max")
    st.write("""
    Pada proses normalisasi data akan diubah menjadi skala data sehingga memiliki
    distribusi yang sama. Proses ini dilakukan untuk mengatasi perbedaan skala pada
    data. Pada dataset penyakit jantung ini, kolom klasifikasi tidak dilakukan normalisasi
    karena terdiri dari dua kelas, yaitu 0 tidak beresiko penyakit jantung dan 1 beresiko
    penyakit jantung. 

    """)
    st.dataframe(data_scaler.head())

# =========================
# MENU PREDIKSI
# =========================
elif menu == "Prediksi":

    st.title("Prediksi Penyakit Jantung")
    st.write(selected_features)
    st.write("Jumlah fitur:", len(selected_features))

    # =========================
    # INPUT USER
    # =========================

    male = st.selectbox("Jenis Kelamin", [0,1])

    age = st.number_input("Umur", 20, 100, 30)

    education = st.selectbox(
        "Education",
        [1,2,3,4]
    )

    currentSmoker = st.selectbox(
        "Current Smoker",
        [0,1]
    )

    cigsPerDay = st.number_input(
        "Cigarettes Per Day",
        0,
        50,
        0
    )

    BPMeds = st.selectbox(
        "BP Meds",
        [0,1]
    )

    prevalentStroke = st.selectbox(
        "Prevalent Stroke",
        [0,1]
    )

    prevalentHyp = st.selectbox(
        "Prevalent Hypertension",
        [0,1]
    )

    diabetes = st.selectbox(
        "Diabetes",
        [0,1]
    )

    totChol = st.number_input(
        "Total Cholesterol",
        100,
        600,
        200
    )

    sysBP = st.number_input(
        "Systolic BP",
        80,
        250,
        120
    )

    diaBP = st.number_input(
        "Diastolic BP",
        50,
        150,
        80
    )

    BMI = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    heartRate = st.number_input(
        "Heart Rate",
        40,
        200,
        80
    )

    glucose = st.number_input(
        "Glucose",
        40,
        400,
        100
    )

    # =========================
    # DATA INPUT
    # =========================

    input_data = pd.DataFrame({
        'male': [male],
        'age': [age],
        'education': [education],
        'currentSmoker': [currentSmoker],
        'cigsPerDay': [cigsPerDay],
        'BPMeds': [BPMeds],
        'prevalentStroke': [prevalentStroke],
        'prevalentHyp': [prevalentHyp],
        'diabetes': [diabetes],
        'totChol': [totChol],
        'sysBP': [sysBP],
        'diaBP': [diaBP],
        'BMI': [BMI],
        'heartRate': [heartRate],
        'glucose': [glucose]
    })

    # =========================
    # NORMALISASI
    # =========================
    # scaler dilatih dengan semua fitur
    input_scaled = scaler.transform(input_data)

    # ubah ke dataframe lagi
    input_scaled = pd.DataFrame(
        input_scaled,
        columns=input_data.columns
    )

    # =========================
    # AMBIL FITUR TERPILIH
    # =========================
    input_selected = input_scaled[selected_features]

    # =========================
    # PREDIKSI
    # =========================
    if st.button("Prediksi"):

        prediction = model.predict(input_selected)

        if prediction[0] == 1:
            st.error("Terindikasi Penyakit Jantung")

        else:
            st.success("Tidak Terindikasi Penyakit Jantung")
# =========================
# COPYRIGHT
# =========================
    st.markdown(
        """
        <div class="copyright">
        © 2026 Klasifikasi Penyakit jantung | by Niken Amalia
        </div>
        """,
    unsafe_allow_html=True
    )

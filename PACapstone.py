import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
import sklearn
import io
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
print(sklearn.__version__)


@st.cache_data
def load_data():
    df = pd.read_csv("Data_CleaningFIX1.csv")
    return df
with open('PACapstonePickle.pkl', 'rb') as f:
    PACapstonePickle = pickle.load(f)
# Fungsi untuk menampilkan halaman tentang
def show_About():
    st.header("Toko Roti Bahari")
    st.write("""
## Business Understanding

### Business objective
Meningkatkan efisiensi operasional dan pengalaman pelanggan Toko Roti Bahari melalui implementasi website desktop yang mengotomatiskan proses penjualan, manajemen inventaris, dan pencatatan transaksi, sehingga dapat meningkatkan pendapatan, produktivitas, dan kompetitivitas usaha.

### Assess situation
Toko Roti Bahari saat ini menghadapi tantangan dalam menjaga efisiensi operasional dan meningkatkan pengalaman pelanggan. Proses penjualan dan manajemen inventaris masih dilakukan secara manual, menyebabkan pemborosan waktu dan tenaga. Selain itu, dengan persaingan pasar yang semakin ketat, Toko Roti Bahari perlu meningkatkan keunggulan kompetitifnya melalui penerapan teknologi yang canggih.

### Tujuan data mining
Mengumpulkan data penjualan dan inventaris yang komprehensif untuk analisis lebih lanjut. Menganalisis pola pembelian pelanggan untuk meningkatkan strategi pemasaran. Menemukan tren produk yang dapat meningkatkan penjualan dan keuntungan. Mengidentifikasi area di mana efisiensi operasional dapat ditingkatkan melalui otomatisasi proses.

### Rencana proyek
Proyek harus terdiri dari beberapa tahap: pengumpulan data, analisis data, pembuatan model, dan implementasi strategi. Setiap tahap harus diberi waktu yang cukup untuk menyelesaikannya dengan baik. Setelah model selesai, strategi berdasarkan wawasan model harus diimplementasikan dan dievaluasi untuk efektivitasnya.
    """)
    st.subheader("Dataset")
    st.dataframe(df)
    
    st.subheader("Metodologi")
    st.write("Metodologi yang digunakan pada proyek data mining ini adalah CRISP-DM yang merupakan singkatan dari Cross-Industry Standard Process for Data Mining. Metodologi ini digunakan karena ")

    st.subheader("Pengembangan Model")
    st.write("""Pada bagian pengembangan model ini fitur yang dipilih untuk digunakan adalah 'Efektivitas_Penjualan' yang merupakan hasil dari penilaian dari empat kolom yaitu, 'Total Pembuatan', 'Sisa barang', 'Total_Produksi_terjual', dan 'Total_Pendapatan' yang telah diubah menjadi rasio melalui min-max scaling yang akan digunakan untuk menilai efektivitas dari penjualan sebuah produk. Model yang digunakan pada pengembangan model ini yaitu, Gaussian Naive Bayes, K-Nearest Neighbor, dan Decision Tree Classifier. Pada bagian evaluasi setelah dilakukannnya pelatihan pada tiga model tersebut K-Nearest Neighbor dan Decision Tree Classifier memiliki nilai yang sempurna pada setiap penilaian yang ditentukan sedangkan Gaussian Naive Bayes memiliki hasil penilaian yang lebih rendah. Setelah dilakukan validasi juga K-Nearest Neighbor dan Decision Tree Classifier memiliki akurasi yang sempurna pada tiap fold nya sedangkan Gaussian Naive Bayes memiliki peningkatan dibandingkan dari fold 1 dan fold 5 nya. Dengan sempurnanya nilai pada model K-Nearest Neighbor dan Decision Tree Classifier dapat terjadi karena adanya overfitting dan dua model tersebut gagal menggeneralisasi ke data yang baru.""")

def show_VisualizationAndInsight(df):
    st.header("Visualisasi dan Insight")
    st.subheader("Chart Frekuensi Produk Dengan Kategori Pendapatan")
    numerical_column = 'Frekuensi'  # Choose the column you want to analyze

    # Function to generate the bar chart
    def plot_bar_chart(data, numerical_column):
        fig, ax = plt.subplots()
        category_counts = data['Pendapatan_Category'].value_counts()
        ax.bar(category_counts.index, category_counts.values)

        # Add labels and title
        ax.set_xlabel('Pendapatan_Category')
        ax.set_ylabel(numerical_column)
        ax.set_title(f'Distribution of {numerical_column} by Pendapatan_Category')
        ax.grid(True)

        return fig

    
    # Display the bar chart in Streamlit
    st.pyplot(plot_bar_chart(df, numerical_column))
    st.subheader("Interpretation")
    st.write("Chart diatas merepresentasikan frekuensi kategori pendapatan dari tiap produk, pada label sumbu x yaitu kategori pendapatan memiliki value rendah, sedang, dan tinggi. Dan pada label sumbu y yaitu Frekuensi memiliki value mulai 0 hingga 500.")

    st.subheader("Insight")
    st.write("Pada chart diatas dari keseluruhan produk dari toko roti ini lebih dari 800 produk memiliki kategori pendapatan yang tinggi dan sedang, sedangkan sekitar 200 an produk memiliki kategori pendapatan yang rendah.")

    st.subheader("Actionable Insight")
    st.write("Dari chart diatas analisi lebih lanjut diperlu untuk menilai mengapa sekitar 200 produk itu memiliki kategori pendapatan yang rendah, agar dapat ditingkatkan untuk meningkatkan penjualan serta pendapatan dari toko tersebut.")

    st.subheader("Chart Top 10 Produk Dengan Pendapatan Rendah")
    # Filter data untuk Pendapatan_Category "rendah"
    top_10_genres = df.groupby('Nama Produk')['Total_Pendapatan'].mean().sort_values(ascending=True).head(10).index
    df_top_10 = df[df['Nama Produk'].isin(top_10_genres)]

    # Create the bar plot without error bars and in descending order
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='Total_Pendapatan', y='Nama Produk', data=df_top_10, ci=None,
                order=top_10_genres,
                ax=ax)
    # Set labels and title
    ax.set_xlabel('Total Pendapatan')
    ax.set_ylabel('Nama Produk')
    ax.set_title('Top 10 Produk dengan Pendapatan Terendah')
    # Show the plot in Streamlit
    st.pyplot(fig)
    st.subheader("Interpretation")
    st.write("Chart diatas menampilkan 10 produk dengan total pendapatan terendah dengan sumbu x yang memiliki nilai mulai dari 0 hingga 100.000 dan sumbu y yang menampilkan nilai dari nama produknya.")

    st.subheader("Insight")
    st.write("Pada chart diatas 10 produk dengan total pendapatan mulai dari pisang goreng dengan total pendapatan terendah diikuti dengan ketan inti, ketan serondeng pedas, bolu  jadul ori potongan, bolu jadul pandan potongan, bolu jadul gulmer potongan, bolu kacang hijau, pandan ceres, roti gula, dan bolu coklat.")

    st.subheader("Actionable Insight")
    st.write("Dari chart diatas untuk meningkatkan total pendapatan dari produk terendah tersebut banyak hal yang harus diperhatikan mulai dari kualitas produk, kualitas bahan mentah yang digunakan, harga, stok, dan aspek lainnya yang mempengaruhi bagaimana produk tersebut bisa memiliki nilai pendapatan yang rendah. Setelah mengetahui aspek mana yang menyebabkan produk tersebut memiliki total pendapatan yang rendah maka hal tersebut dapat dijadikan acuan untuk diperbaiki agar meningkatkan pendapatan dari produk tersebut agar menjadi lebih maksimal.")

    # Filter data untuk Pendapatan_Category "rendah"
    top_10_genres = df.groupby('Nama Produk')['Total_Pendapatan'].mean().sort_values(ascending=True).head(10).index
    df_top_10 = df[df['Nama Produk'].isin(top_10_genres)]

    # Create the bar plot without error bars and in descending order
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='Sisa barang', y='Nama Produk', data=df_top_10, ci=None,
                order=top_10_genres,
                ax=ax)
    # Set labels and title
    ax.set_xlabel('Sisa barang')
    ax.set_ylabel('Nama Produk')
    ax.set_title('Sisa Barang dari Top 10 Produk dengan Pendapatan Terendah')
    # Show the plot in Streamlit
    st.pyplot(fig)
    st.subheader("Interpretation")
    st.write("Chart diatas menampilkan sisa barang dari 10 produk dengan total pendapatan terendah dengan sumbu x yang memiliki nilai mulai dari 0 hingga 3.0 dan sumbu y yang menampilkan nilai dari nama produknya.")

    st.subheader("Insight")
    st.write("Pada chart diatas sisa barang dari 10 produk dengan total pendapatan mulai dari bolu kacang hijau dan pandan ceres dengan nilai sisa barang tertinggi diikuti dengan pisang goreng, ketan inti, ketan serondeng pedas, bolu jadul ori potongan, bolu jadul pandan potongan, bolu jadul gulmer potongan, roti gula, dan bolu coklat.")

    st.subheader("Actionable Insight")
    st.write("Dari chart diatas dengan nilai sisa barang yang paling tinggi kurang dari 3 jika diasumsikan bahwa konsumen ingin membeli produk tersebut dengan kuantitas lebih dari pada 3 maka total pembuatan dari 10 produk tersebut dapat ditingkatkan secara berkala sembari melihat apabila terjadi peningkatan terhadap sisa barang maka asumsi ini dapat ditolak mengingat bahwa kualitas dari produk tersebut lah yang menjadi kendala dari rendahnya pendapatan yang dihasilkan dari produk tersebut.")


    # Filter data untuk Pendapatan_Category "rendah" dan mengkecualikan produk "Air Mineral" dan "kotak"
    top_10_genres = df[df['Nama Produk'] != 'Air Mineral']  # Mengecualikan produk "Air Mineral"
    top_10_genres = top_10_genres[top_10_genres['Nama Produk'] != 'kotak']  # Mengecualikan produk "kotak"
    top_10_genres = top_10_genres.groupby('Nama Produk')['Total_Produksi_terjual'].mean().sort_values(ascending=False).head(10).index
    df_top_10 = df[df['Nama Produk'].isin(top_10_genres)]

    # Create the bar plot without error bars and in descending order
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x='Total_Produksi_terjual', y='Nama Produk', data=df_top_10, ci=None,
                order=top_10_genres,
                ax=ax)
    # Set labels and title
    ax.set_xlabel('Total Produksi Terjual')
    ax.set_ylabel('Nama Produk')
    ax.set_title('Top 10 Produk dengan Total Produksi Terjual')
    # Show the plot in Streamlit
    st.pyplot(fig)
    st.subheader("Interpretation")
    st.write("Chart diatas menampilkan 10 produk dengan total produksi terjual tertinggi dengan sumbu x yang memiliki nilai mulai dari 0 hingga 160 dan sumbu y yang menampilkan nama produknya.")

    st.subheader("Insight")
    st.write("Pada chart diatas 10 produk dengan total produksi terjual tertinggi mulai dari bingka gula merah, cantik manis, ilat sapi merah, bingka pisang, ilat sapi putih, untuk-untuk inti, risol kentang, bolu kukus ungu, bolu kukus pink, hingga bolu kukus coklat.")

    st.subheader("Actionable Insight")
    st.write("Dari chart diatas hal yang dapat dilakukan untuk meningkatkan penjualan dari toko adalah dengan menaruh 10 produkt tersebut di etalase paling depan pada toko, pada website 10 produk tersebut dapat dijadikan produk yang akan mengangkat promosi dari toko nantinya dengan menampilkan produk-produk tersebut pada bagian paling depan pada halaman website nantinya.")
    st.write("Terkhusus untuk bolu kukus yang memiliki tiga varian yaitu, ungu, pink, dan coklat dapat dilihat bahwa tiga varian tersebut memiliki total terjual yang tinggi untuk produk yang sama dengan jenis yang berbeda, hal yang dapat dilakukan adalah dengan membuat inovasi varian baru dari bolu kukus tersebut untuk menambah pilihan bagi konsumen lebih banyak lagi. Hal ini dapat meningkatkan penjualan dikarenakan dari awal bolu kukus memiliki tingkat terjual yang tinggi dan konsisten meskipun memiliki tiga varian yang berbeda.")

def predict_cancellation():
    
    def predict_efektivitas_penjualan(new_row):
        df = pd.read_csv('Data_CleaningFIX1.csv')
        # Menambahkan data baru ke DataFrame
        df.loc[len(df)] = new_row  # Menambahkan row baru ke DataFrame
        
        numeric_columns = ['Harga/satuan', 'Total Pembuatan', 'Sisa barang', 'Total_Produksi_terjual', 'Total_Pendapatan']

        # Normalisasi menggunakan min-max scaling
        for column in numeric_columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        df = df.drop(columns=['Nama Produk','tanggal','Minggu Ke-', 'Pendapatan_Category'])
        def map_to_category(value, threshold):
            if value < threshold:
                return 0
            else:
                return 1

        # Menentukan threshold untuk mapping
        threshold = 0.5  # Misalnya, kita akan menentukan rasio di atas 0.5 sebagai "tinggi"

        # Melakukan mapping untuk setiap kolom rasio
        for column in df.columns:  # Mengiterasi melalui semua kolom # Mengabaikan kolom 'Nama Produk'
                df[column] = df[column].apply(lambda x: map_to_category(x, threshold))
        # Prepare input for prediction
        input_data = df.iloc[-1][['Harga/satuan', 'Total Pembuatan', 'Sisa barang', 'Total_Produksi_terjual', 'Total_Pendapatan']].values.reshape(1, -1)  # Mengambil row terakhir dari DataFrame dan mengubahnya menjadi list
        df = df.drop(df.index[-1])
        # Perform prediction
        EfektivitasPenjualan = PACapstonePickle.predict(input_data)

        # Map prediction to "low" or "high"
        if EfektivitasPenjualan == 0:
            prediction_result = "rendah"
        else:
            prediction_result = "tinggi"
        # prediction_result = "rendah" if EfektivitasPenjualan[0] == 1 else "tinggi"
        return prediction_result

    # Menampilkan input dari pengguna
    st.title('Data Mining Efektivitas Penjualan Predict')
    HargaSatuan = st.text_input('Input nilai HargaSatuan')
    TotalPembuatan = st.text_input('Input nilai TotalPembuatan')
    SisaBarang = st.text_input('Input nilai SisaBarang')
    Total_Produksi_terjual = st.text_input('Input nilai Total_Produksi_terjual')
    Total_Pendapatan = st.text_input('Input nilai Total_Pendapatan')

    # Memeriksa apakah tombol prediksi ditekan
    if st.button('Prediksi Efektivitas Penjualan'):
        # Memeriksa apakah semua input terisi
        if HargaSatuan and TotalPembuatan and SisaBarang and Total_Produksi_terjual and Total_Pendapatan:
            # Memasukkan nilai inputan ke dalam DataFrame df
            new_row = {
                'Harga/satuan': float(HargaSatuan),
                'Total Pembuatan': float(TotalPembuatan),
                'Sisa barang': float(SisaBarang),
                'Total_Produksi_terjual': float(Total_Produksi_terjual),
                'Total_Pendapatan': float(Total_Pendapatan)
            }

            # Memanggil fungsi prediksi
            prediction_result = predict_efektivitas_penjualan(new_row)
            
            # Menampilkan hasil prediksi
            st.write(f'Predicted Efektivitas Penjualan: {prediction_result}')
        else:
            st.error("Mohon isi semua nilai input sebelum melakukan prediksi.")
# Assuming df is your DataFrame
# df = pd.read_csv('hotel_data.csv')  # Load your data here
# predict_cancellation(df)

# Memuat data
df = load_data()

nav_options = {
    "About": lambda: show_About(),
    "Visualisasi dan Insight": lambda: show_VisualizationAndInsight(df),
    # "Relationships": lambda: show_hubungan(df),
    # "Comparisons": lambda: show_Perbandingan(df),
    # "Compositions": lambda: show_Komposisi(df),
    "Model Predicting": lambda: predict_cancellation()
}

# Menampilkan sidebar
st.sidebar.title("Analisis Toko Roti Bahari Menggunakan Model CRISP-DM")
selected_page = st.sidebar.radio("Menu", list(nav_options.keys()))

# Menampilkan halaman yang dipilih
nav_options[selected_page]()

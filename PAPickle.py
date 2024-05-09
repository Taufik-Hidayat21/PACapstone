import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
import pickle

# Read the data
df = pd.read_csv('Data_CleaningFIX1.csv')
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


df['Efektivitas Penjualan'] = 1
def add_new_column(row):
    if (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 1 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 1 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 1 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 1):
        return 0
    elif (row['Total Pembuatan'] == 1 and row['Sisa barang'] == 1 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 1 and
          row['Total_Produksi_terjual'] == 1 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 1 and row['Total_Pendapatan'] == 1):
        return 0
    elif (row['Total Pembuatan'] == 1 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 1):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 1 and
          row['Total_Produksi_terjual'] == 1 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 1 and row['Sisa barang'] == 0 and
          row['Total_Produksi_terjual'] == 1 and row['Total_Pendapatan'] == 0):
        return 0
    elif (row['Total Pembuatan'] == 0 and row['Sisa barang'] == 1 and
          row['Total_Produksi_terjual'] == 0 and row['Total_Pendapatan'] == 1):
        return 0
    return 1

# Menambahkan kolom baru ke DataFrame
df['Efektivitas Penjualan'] = df.apply(add_new_column, axis=1)


# Split the data into features and target variable
X = df.drop(columns=['Efektivitas Penjualan'])  # Features
y = df['Efektivitas Penjualan']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussianNB()  # Use Gaussian Naive Bayes
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('PACapstonePickle.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'PACapstonePickle.pkl'")

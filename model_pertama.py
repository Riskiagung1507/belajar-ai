from sklearn.linear_model import LinearRegression

# Contoh data: 
# Kolom pertama = Jumlah Jam Belajar
# Kolom kedua = Nilai Ujian yang Didapat
jam_belajar = [[1], [2], [3], [4], [5]]
nilai_ujian = [50, 60, 70, 80, 90]

# 1. Membuat "otak" model AI-nya (Linear Regression)
model = LinearRegression()

# 2. Melatih model menggunakan data di atas (Proses Training)
model.fit(jam_belajar, nilai_ujian)

# 3. Mencoba menebak: Kalau seseorang belajar selama 6 jam, berapa nilainya?
prediksi_nilai = model.predict([[6]])

print("Prediksi nilai jika belajar 6 jam adalah:", prediksi_nilai[0])
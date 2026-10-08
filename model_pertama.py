from sklearn.linear_model import LinearRegression

jam_belajar = [[1], [2], [3], [4], [5]]
nilai_ujian = [50, 60, 70, 80, 90]

model = LinearRegression()

model.fit(jam_belajar, nilai_ujian)

prediksi_nilai = model.predict([[6]])

print("Prediksi nilai jika belajar 6 jam adalah:", prediksi_nilai[0])
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Memuat dataset gambar angka (dari 0 sampai 9)
digits = load_digits()

# Menampilkan informasi dasar data
print("Total data gambar angka:", len(digits.data))

# 2. Membagi data untuk latihan dan ujian
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# 3. Membuat model AI (Random Forest Classifier - cocok untuk pola kompleks)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Melatih model dengan data gambar latihan
model.fit(X_train, y_train)

# 5. Menguji akurasi model
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

print("Berhasil melatih AI pengenalan gambar!")
print("Tingkat akurasi AI mengenali angka adalah:", akurasi * 100, "%")
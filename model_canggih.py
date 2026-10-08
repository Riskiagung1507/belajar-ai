from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Memuat data bunga Iris
iris = load_iris()
X = iris.data  # Fitur (ukuran panjang & lebar kelopak/mahkota)
y = iris.target  # Label jenis bunganya (0, 1, atau 2)

# 2. Membagi data: Sebagian untuk belajar (training), sebagian untuk ujian (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Membuat "otak" AI-nya (KNN Classifier)
model = KNeighborsClassifier(n_neighbors=3)

# 4. Melatih model dengan data belajar
model.fit(X_train, y_train)

# 5. Menguji akurasi model menggunakan data ujian
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

print("Berhasil melatih model klasifikasi!")
print("Akurasi model AI ini adalah:", akurasi * 100, "%")
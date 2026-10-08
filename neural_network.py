import torch
import torch.nn as nn
import torch.optim as optim

# 1. Membuat data sederhana (Contoh: 4 data latihan dengan 2 fitur)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
# Target/jawaban yang diinginkan (Logika XOR sederhana)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# 2. Merancang "Otak Buatan" (Neural Network sederhana dengan 1 Hidden Layer)
class ModelOtakBuatan(nn.Module):
    def __init__(self):
        super(ModelOtakBuatan, self).__init__()
        self.layer1 = nn.Linear(2, 4)  # 2 input ke 4 neuron tersembunyi
        self.relu = nn.ReLU()          # Fungsi aktivasi supaya AI bisa berpikir non-linear
        self.layer2 = nn.Linear(4, 1)  # 4 neuron ke 1 output
        self.sigmoid = SigmoidCustom = nn.Sigmoid() # Supaya hasil antara 0 dan 1

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

model = ModelOtakBuatan()

# 3. Menentukan cara belajar (Loss function & Optimizer)
criterion = nn.BCELoss() # Menghitung tingkat error
optimizer = optim.SGD(model.parameters(), lr=0.1) # Kecepatan belajar (learning rate)

# 4. Proses Training (Belajar berulang-ulang sebanyak 1000 kali)
print("Memulai proses training Neural Network...")
for epoch in range(1000):
    optimizer.zero_grad()
    prediksi = model(X)
    loss = criterion(prediksi, y)
    loss.backward()
    optimizer.step()

print("Training selesai!")

# 5. Menguji hasil belajar AI
print("\nHasil Prediksi AI setelah belajar:")
with torch.no_grad():
    print(model(X))
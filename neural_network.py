import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)


class ModelOtakBuatan(nn.Module):
    def __init__(self):
        super(ModelOtakBuatan, self).__init__()
        self.layer1 = nn.Linear(2, 4) 
        self.relu = nn.ReLU()          
        self.layer2 = nn.Linear(4, 1) 
        self.sigmoid = SigmoidCustom = nn.Sigmoid() 

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

model = ModelOtakBuatan()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Memulai proses training Neural Network...")
for epoch in range(10000):
    optimizer.zero_grad()
    prediksi = model(X)
    loss = criterion(prediksi, y)
    loss.backward()
    optimizer.step()

print("Training selesai!")

print("\nHasil Prediksi AI setelah belajar:")
with torch.no_grad():
    print(model(X))
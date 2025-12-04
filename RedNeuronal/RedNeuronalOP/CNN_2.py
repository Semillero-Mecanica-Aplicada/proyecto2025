## Elaborado por: Semillero de Mecánica Aplicada - EAFIT

## Red neuronal convolucional (CNN)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Creación de dataset

class QRDatasetNPZ(Dataset):
    def __init__(self, img_path, resp_path):
        self.X = np.load(img_path)['arr_0'][0:50000]  
        self.Y = np.load(resp_path)['results']
        assert len(self.X) == len(self.Y), "Error: longitudes distintas"
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # Añadir canal
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y

# --- Uso ---
dataset = QRDatasetNPZ('images0_125.npz', 'Resultados0_50.npz')

# Normalizar coeficientes
dataset.Y = dataset.Y/(12e9)
mask = ~np.isnan(dataset.Y).any(axis=1)
dataset.X = dataset.X[mask]
dataset.Y = dataset.Y[mask]

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training and testing data
# 90% entrenamiento, 10% test
train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Creación del modelo CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,5,1, padding=2)
        self.conv2 = nn.Conv2d(16,16,5,1, padding=2)
        self.conv3 = nn.Conv2d(16,32,5,1, padding=2)
        self.conv4 = nn.Conv2d(32,32,5,1, padding=2)
        # Fully connected layer
        self.fc1 = nn.Linear(32*4*4, 6) # FC(32*8*8x6)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        # Second pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        # Third pass
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        # Fourth pass
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        # Re-View to flatten it out
        X = X.view(-1, 32*4*4) # Negative one so we can vary the batch size. # Flatten to (batch_size, 32*8*8)

        # Fully connected layers
        X = self.fc1(X)
        
        return X

# Create an Instance of our Model
torch.manual_seed(2) # A random number


# Training the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model = model.to(device)


criterion = nn.MSELoss()           
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
loss_history = []  # para almacenar la historia de pérdidas

for epoch in range(epochs):
    running_loss = 0.0

    for X_train, Y_train in train_loader:
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        # Forward
        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "modelo_final_CNN2.pth")


# Evaluación en Test
model.eval()
test_loss = 0

with torch.no_grad():
    for X_test, Y_test in test_loader:
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        y_pred = model(X_test)
        loss = criterion(y_pred, Y_test)
        test_loss += loss.item()



# Guardar epochs, loss_history y Evaluación en un archivo npz
np.savez(
    "loss_CNN2.npz",
    epochs=list(range(len(loss_history))),
    loss=loss_history,
    test_error=test_loss / len(test_loader)
)

print(f"Test Loss: {test_loss / len(test_loader):.6f}")
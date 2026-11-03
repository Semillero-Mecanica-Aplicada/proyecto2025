## Elaborado por: Semillero de Mecánica Aplicada - EAFIT

## Red neuronal convolucional inversa (CNN 4 invertida)
## Entrada: 6 coeficientes [C11, C22, C66, C12, C26, C16] -> Salida: imagen binaria 64x64

import os
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

# Semilla antes del random_split para que la partición sea reproducible
torch.manual_seed(2)

# Rutas relativas a este archivo (funciona sin importar desde qué carpeta se ejecute)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # RedNeuronal/RedNeuronalOP
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))      # proyecto2025


# Creación de dataset (roles invertidos: X = coeficientes, Y = imagen)

class QRDatasetNPZ(Dataset):
    def __init__(self, img_path, resp_path):
        self.X = np.load(img_path)['arr_0'][0:50000]
        self.Y = np.load(resp_path)['results']
        assert len(self.X) == len(self.Y), "Error: longitudes distintas"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.Y[idx], dtype=torch.float32)               # Coeficientes (6,)
        y = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # Imagen (1,64,64)
        return x, y

# --- Uso ---
dataset = QRDatasetNPZ(os.path.join(REPO_DIR, 'Generacion de imagenes', 'images0_125.npz'),
                       os.path.join(REPO_DIR, 'Resultados', 'Resultados0_50.npz'))

# Normalizar coeficientes (misma escala que CNN_4 para poder hacer la verificación de ciclo)
# Nota: C26 y C16 son ~0 frente a C11/C22; si la red no aprende, probar StandardScaler
# ajustado solo con el conjunto de entrenamiento.
dataset.Y = dataset.Y/(12e9)
mask = ~np.isnan(dataset.Y).any(axis=1)
dataset.X = dataset.X[mask]
dataset.Y = dataset.Y[mask]

# Training and testing data
# 90% entrenamiento, 10% test
train_size = int(0.9 * len(dataset))
test_size  = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Creación del modelo CNN inverso (espejo de CNN_4)
class InverseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected layers (inversas de fc2: 1000 -> 6 y fc1: 32*8*8 -> 1000)
        # Sin ReLU entre fc2 y fc1, igual que en CNN_4
        self.fc2 = nn.Linear(6, 1000)
        self.fc1 = nn.Linear(1000, 32*8*8)
        # MaxPool no tiene inversa exacta: se reemplaza por Upsample x2 + Conv2d
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(32,16,5,1, padding=2)  # inversa de conv3 (16->32)
        self.conv2 = nn.Conv2d(16,16,5,1, padding=2)  # inversa de conv2 (16->16)
        self.conv1 = nn.Conv2d(16,1,5,1, padding=2)   # inversa de conv1 (1->16)

    def forward(self, X):
        # Fully connected layers
        X = self.fc2(X)
        X = F.relu(self.fc1(X))
        # Re-View to un-flatten it
        X = X.view(-1, 32, 8, 8)
        # Third pass (inverso): 8x8 -> 16x16
        X = self.up(X)
        X = F.relu(self.conv3(X))
        # Second pass (inverso): 16x16 -> 32x32
        X = self.up(X)
        X = F.relu(self.conv2(X))
        # First pass (inverso): 32x32 -> 64x64
        X = self.up(X)
        X = self.conv1(X)  # Logits, sin ReLU (sigmoid va en la pérdida / inferencia)

        return X


# Modelo directo de CNN_4 (copiado para la verificación de ciclo imagen -> coeficientes)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,5,1, padding=2)
        self.conv2 = nn.Conv2d(16,16,5,1, padding=2)
        self.conv3 = nn.Conv2d(16,32,5,1, padding=2)
        self.fc1 = nn.Linear(32*8*8, 1000)
        self.fc2 = nn.Linear(1000, 6)

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2, 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2, 2)
        X = F.max_pool2d(F.relu(self.conv3(X)), 2, 2)
        X = X.view(-1, 32*8*8)
        X = self.fc1(X)
        return self.fc2(X)


# Training the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InverseCNN()
model = model.to(device)


criterion = nn.BCEWithLogitsLoss()  # Salida binaria (0/1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50  # Prueba corta; subir a 1000 como en CNN_4 cuando todo funcione
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

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "modelo_final_CNN4_inv.pth"))


# Evaluación en Test
model.eval()
test_loss = 0
correct_pixels = 0
total_pixels = 0
intersection = 0
union = 0

with torch.no_grad():
    for X_test, Y_test in test_loader:
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        y_pred = model(X_test)
        loss = criterion(y_pred, Y_test)
        test_loss += loss.item()

        # Imagen binarizada
        y_bin = (torch.sigmoid(y_pred) > 0.5).float()
        correct_pixels += (y_bin == Y_test).sum().item()
        total_pixels += Y_test.numel()
        intersection += (y_bin * Y_test).sum().item()
        union += ((y_bin + Y_test) > 0).sum().item()

pixel_acc = correct_pixels / total_pixels
iou = intersection / union if union > 0 else 0.0
print(f"Test BCE: {test_loss / len(test_loader):.6f} - Pixel acc: {pixel_acc:.4f} - IoU: {iou:.4f}")


# Verificación de ciclo: coeficientes -> imagen predicha -> CNN_4 -> coeficientes
# (la imagen puede diferir de la real y aun así tener los mismos coeficientes)
cycle_error = np.nan
forward_path = None
for folder in (SCRIPT_DIR, REPO_DIR):  # CNN_4 guarda el modelo en la carpeta desde donde se ejecutó
    if os.path.exists(os.path.join(folder, "modelo_final_CNN4.pth")):
        forward_path = os.path.join(folder, "modelo_final_CNN4.pth")
        break
if forward_path is not None:
    forward_model = CNN().to(device)
    forward_model.load_state_dict(torch.load(forward_path, map_location=device))
    forward_model.eval()
    mse = nn.MSELoss()
    cycle_loss = 0
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test = X_test.to(device)
            y_bin = (torch.sigmoid(model(X_test)) > 0.5).float()
            cycle_loss += mse(forward_model(y_bin), X_test).item()
    cycle_error = cycle_loss / len(test_loader)
    print(f"Error de ciclo (MSE coeficientes): {cycle_error:.6f}")
else:
    print("No se encontró modelo_final_CNN4.pth; se omite la verificación de ciclo")


# Comparación visual de algunas imágenes de test
X_s, Y_s = next(iter(test_loader))
with torch.no_grad():
    prob = torch.sigmoid(model(X_s.to(device))).cpu()

n = 5
fig, axes = plt.subplots(3, n, figsize=(2.5*n, 7.5))
for i in range(n):
    axes[0, i].imshow(Y_s[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].imshow(prob[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[2, i].imshow(prob[i, 0] > 0.5, cmap='gray', vmin=0, vmax=1)
    for ax in axes[:, i]:
        ax.axis('off')
axes[0, 0].set_title('Real', loc='left')
axes[1, 0].set_title('Probabilidad', loc='left')
axes[2, 0].set_title('Binarizada', loc='left')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "comparacion_CNN4_inv.png"), dpi=150)


# Guardar epochs, loss_history y Evaluación en un archivo npz
np.savez(
    os.path.join(SCRIPT_DIR, "loss_CNN4_inv.npz"),
    epochs=list(range(len(loss_history))),
    loss=loss_history,
    test_error=test_loss / len(test_loader),
    pixel_acc=pixel_acc,
    iou=iou,
    cycle_error=cycle_error
)

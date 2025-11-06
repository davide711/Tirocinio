# Test su Fed-Heart-Disease (fhd) con sistema centralizzato


# Importiamo le librerie necessarie
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

import centralized_utils as cu

# ------------------ 1. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Creazione del modello centrale
central_model = cu.NeuralNetwork().to(device)

# ------------------------ 2. PREPARAZIONE DEI DATI --------------------------
from flamby.datasets.fed_heart_disease import FedHeartDisease

client_nns = 4

train_dataloaders = [
            torch.utils.data.DataLoader(
                FedHeartDisease(center = i, train = True, pooled = False),
                batch_size = 64,
                shuffle = True,
                num_workers = 0
            )
            for i in range(client_nns)
        ]

client_loaders = [train_dataloaders[0], train_dataloaders[1], train_dataloaders[2], train_dataloaders[3]]

client_datasets = [loader.dataset for loader in client_loaders]

merged_dataset = ConcatDataset(client_datasets)

merged_loader = DataLoader(merged_dataset, batch_size=64, shuffle=True)

test_dataloaders = torch.utils.data.DataLoader(
                FedHeartDisease(train = False, pooled = False),
                batch_size = 64,
                shuffle = True,
                num_workers = 0
            )

# ---------------------------- 3. PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate = 0.05

# Inizializzazione della funzione cross-entropy
loss_fn = nn.BCEWithLogitsLoss()

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model.parameters(), lr=learning_rate)

# Ripetiamo train e test loop per 30 epoche, stampando i risultati
epochs = 30

# Impostiamo un valore di patience (in numero di epoche senza miglioramento)
patience = 2

rounds_array = []
accuracy_array = []

for t in range(epochs):
    print(f"Round {t+1}\n-------------------------------")
    cu.train_loop(merged_loader, central_model, loss_fn, optimizer)
    rounds_array.append(t+1)
    if(cu.test_loop(test_dataloaders, central_model, loss_fn, accuracy_array, patience) == 1):
        break
    

# Plot dei risultati
cu.plot(rounds_array, accuracy_array)

print("Done!")

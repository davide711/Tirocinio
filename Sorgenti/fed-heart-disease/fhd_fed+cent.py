# Test su fhd con entrambi i modelli insieme

# Importiamo le librerie necessarie
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch import nn

import federated_utils as fu
import centralized_utils as cu

import matplotlib.pyplot as plt

# ------------------ 1. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Creazionie del modello centrale, sia per federato che per centralizzato
central_model_fed = cu.NeuralNetwork().to(device)
central_model_cent = fu.NeuralNetwork().to(device)

# ------------------------ 2. PREPARAZIONE DEI DATI --------------------------
from flamby.datasets.fed_heart_disease import FedHeartDisease

client_nns = 4

# Dataloaders già suddivisi -> da usare in federato
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

# Dataloader unito -> da usare in centralizzato
merged_loader = DataLoader(merged_dataset, batch_size=64, shuffle=True)

# Dataloader di test unito -> da usare per entrambi
test_dataloaders = torch.utils.data.DataLoader(
                FedHeartDisease(train = False, pooled = False),
                batch_size = 64,
                shuffle = True,
                num_workers = 0
            )

# ---------------------------- 3. PROVA DELLA RETE ---------------------------------

# Inizializzazione della funzione cross-entropy
loss_fn = nn.BCEWithLogitsLoss()

# ----------------- 3.1. Training e test federato ------------------

# Definizione iperparametri
learning_rate_fed = 0.05
patience = 3
client_epochs = 1

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model_fed.parameters(), lr=learning_rate_fed)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
rounds = 10

rounds_array_fed = []
accuracy_array_fed = []

for t in range(rounds):
    print(f"Round {t+1}\n-------------------------------")
    fu.fed_train_loop(central_model_fed, train_dataloaders, loss_fn, optimizer, client_nns, client_epochs)
    fu.test_loop(test_dataloaders, central_model_fed, loss_fn, accuracy_array_fed, patience)
    rounds_array_fed.append(t+1)

# ----------------- 3.2. Training e test centralizzato -------------

# Definizione iperparametri
learning_rate_cent = 0.08

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model_cent.parameters(), lr=learning_rate_cent)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
epochs = 10

rounds_array_cent = []
accuracy_array_cent = []

for t in range(epochs):
    print(f"Round {t+1}\n-------------------------------")
    cu.train_loop(merged_loader, central_model_cent, loss_fn, optimizer)
    cu.test_loop(test_dataloaders, central_model_cent, loss_fn, accuracy_array_cent, patience)
    rounds_array_cent.append(t+1)

# --------------- 4. PLOT DEI RISULTATI -----------------

plt.plot(rounds_array_fed, accuracy_array_fed, marker='o', label="Federato, lr=" + str(learning_rate_fed))
plt.plot(rounds_array_cent, accuracy_array_cent, marker='s', label="Centralizzato, lr=" + str(learning_rate_cent))
plt.xlabel("Round")
plt.ylabel("Accuracy")   
plt.title("Accuracy per round")
plt.grid(True)
plt.legend()
plt.savefig("accuracy_plot1.png")
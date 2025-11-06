# Test su Fed-Heart-Disease (fhd) con sistema federato


# Importiamo le librerie necessarie
import torch
from torch import nn
import federated_utils as fu


# ----------- CREAZIONE DELLA RETE NEURALE CENTRALE --------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")                     

# Creazionie del modello centrale
central_model = fu.NeuralNetwork().to(device)

# ------------------ IMPORTAZIONE DEI DATI -----------------------------

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

test_dataloaders = torch.utils.data.DataLoader(
                FedHeartDisease(train = False, pooled = False),
                batch_size = 64,
                shuffle = True,
                num_workers = 0
            )

# ------------------- PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate = 0.1
patience = 3
client_epochs = 1

# Inizializzazione della funzione cross-entropy
loss_fn = nn.BCEWithLogitsLoss()

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model.parameters(), lr=learning_rate)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
rounds = 10

rounds_array = []
accuracy_array = []

for t in range(rounds):
    print(f"Round {t+1}\n-------------------------------")
    fu.fed_train_loop(central_model, train_dataloaders, loss_fn, optimizer, client_nns, client_epochs)
    fu.test_loopw(test_dataloaders, central_model, loss_fn, accuracy_array, patience)
    rounds_array.append(t+1)

# Plot dei risultati
fu.plot(rounds_array, accuracy_array)

print("Done!")
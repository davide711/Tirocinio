# Importiamo le librerie necessarie
import torch
from torch import nn

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import fed_utils_medmnist as fu
import cent_utils_medmnist as cu

# ------------------ 1. DOWNLOAD E PREPARAZIONE DEI DATI -------------------------------------

from medmnist import OrganSMNIST

# Dataset di training (train = True), con immagini trasformate in tensori e etichette one-hot
training_data = OrganSMNIST(
    root='data',            
    split='train',
    download=True,          
    transform=ToTensor(),   
)

# Dataset di validation
validation_data = OrganSMNIST(
    root="data",
    split='val',
    download=True,
    transform=ToTensor(),
)

# Dataset di test (train = False)
test_data = OrganSMNIST(
    root="data",
    split='test',
    download=True,
    transform=ToTensor(),
)

# Definizione del numero di reti client
client_nns = 5

# Suddivido il dataset in due modi
iid_loaders = fu.split_dataset_iid(training_data, client_nns)
non_iid_loaders = fu.split_dataset_non_iid(training_data, client_nns)

print("\nSuddivisione IID:")
for i, loader in enumerate(iid_loaders):
    unique_labels = set()
    for _, labels in loader:
        # Se labels è un tensore, lo convertiamo in lista
        labels_list = labels.tolist()
        # Appiattiamo eventuali liste annidate
        for item in labels_list:
            if isinstance(item, list):
                unique_labels.update(item)  # aggiungiamo singoli elementi
            else:
                unique_labels.add(item)
    print(f"Labels uniche client {i+1}: {sorted(unique_labels)}")

print("\nSuddivisione non-IID")

for i, loader in enumerate(non_iid_loaders):
    unique_labels = set()
    for _, labels in loader:
        # Se labels è un tensore, lo convertiamo in lista
        labels_list = labels.tolist()
        # Appiattiamo eventuali liste annidate
        for item in labels_list:
            if isinstance(item, list):
                unique_labels.update(item)  # aggiungiamo singoli elementi
            else:
                unique_labels.add(item)
    print(f"Labels uniche client {i+1}: {sorted(unique_labels)}")

print("\n")
centralized_loader = test_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Il dataset di validation non deve essere diviso
validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

# Il dataset di test non deve essere diviso
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# ------------------ 2. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Creazione dei due modelli centrali
central_model_iid = fu.NeuralNetwork().to(device)
central_model_non_iid = fu.NeuralNetwork().to(device)

# Creazione del modello centralizzato
centralized_model = cu.NeuralNetwork().to(device)

# ---------------------------- 3. PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate_iid = 0.2
learning_rate_non_iid = 0.04
learning_rate_centralized = 0.2
local_epochs = 1
patience = 3

# Inizializzazione della funzione cross-entropy
loss_fn = nn.CrossEntropyLoss()

# Definizione di un ottimizzatore (considero quello del caso iid)
iid_optimizer = torch.optim.SGD(central_model_iid.parameters(), lr=learning_rate_iid)
non_iid_optimizer = torch.optim.SGD(central_model_non_iid.parameters(), lr=learning_rate_non_iid)
centralized_optimizer = torch.optim.SGD(centralized_model.parameters(), lr=learning_rate_centralized)

# Ripetiamo train e test loop per 10 rounds/epoche, stampando i risultati
rounds = 10
rounds_array = []

accuracy_array_iid = []
accuracy_array_non_iid = []
accuracy_array_centralized = []

loss_array_iid = []
loss_array_non_iid = []
loss_array_centralized = []

# Flag per individuare early stopping (es)
es_cent = 0
es_iid = 0
es_non_iid = 0

# Early stopping attivo o no
early_stopping_switch = 1

if(early_stopping_switch == 1):
    for t in range(rounds):
        print(f"Round {t+1}\n-------------------------------")

        if(es_iid == 0):
            fu.fed_train_loop(central_model_iid, iid_loaders, loss_fn, iid_optimizer, client_nns, local_epochs)
        else:
            accuracy_array_iid.append(np.nan)
            loss_array_iid.append(np.nan)

        
        if(es_non_iid == 0):
            fu.fed_train_loop(central_model_non_iid, non_iid_loaders, loss_fn, non_iid_optimizer, client_nns, local_epochs)
        else:
            accuracy_array_non_iid.append(np.nan)
            loss_array_non_iid.append(np.nan)
        

        if(es_cent == 0):
            cu.train_loop(centralized_loader, centralized_model, loss_fn, centralized_optimizer)
        else:
            accuracy_array_centralized.append(np.nan)
            loss_array_centralized.append(np.nan)

        if(es_iid == 0):
            print("---IID---\n")
            if(fu.validation_loop(validation_dataloader, central_model_iid, loss_fn, accuracy_array_iid, patience, loss_array_iid) == 1):
                es_iid = 1
                print("IID: Early stopping")

        if(es_non_iid == 0):
            print("---NON IID---\n")
            if(fu.validation_loop(validation_dataloader, central_model_non_iid, loss_fn, accuracy_array_non_iid, patience, loss_array_non_iid) == 1):
                es_non_iid = 1
                print("non-IID: Early stopping")

        if(es_cent == 0):
            print("---CENTRALIZZATO---\n")
            if(cu.validation_loop(validation_dataloader, centralized_model, loss_fn, accuracy_array_centralized, patience, loss_array_centralized) == 1):
                es_cent = 1
                print("centralizzato: Early stopping")

        rounds_array.append(t+1)

if(early_stopping_switch == 0):
    for t in range(rounds):

        print(f"Round {t+1}\n-------------------------------")

        fu.fed_train_loop(central_model_iid, iid_loaders, loss_fn, iid_optimizer, client_nns, local_epochs)

        fu.fed_train_loop(central_model_non_iid, non_iid_loaders, loss_fn, non_iid_optimizer, client_nns, local_epochs)

        cu.train_loop(centralized_loader, centralized_model, loss_fn, centralized_optimizer)

        print("---IID---\n")
        fu.validation_loop(validation_dataloader, central_model_iid, loss_fn, accuracy_array_iid, patience, loss_array_iid)

        print("---NON IID---\n")
        fu.validation_loop(validation_dataloader, central_model_non_iid, loss_fn, accuracy_array_non_iid, patience, loss_array_non_iid)

        print("---CENTRALIZZATO---\n")
        cu.validation_loop(validation_dataloader, centralized_model, loss_fn, accuracy_array_centralized, patience, loss_array_centralized)

        rounds_array.append(t+1)

print("\nErrore sul dataset di test\n---------------")
print("---IID---\n")
fu.test_loop(test_dataloader, central_model_iid, loss_fn)
print("---NON IID---\n")
fu.test_loop(test_dataloader, central_model_non_iid, loss_fn)
print("---CENTRALIZZATO---\n")
cu.test_loop(test_dataloader, centralized_model, loss_fn)

# Plot dell'accuracy
titolo = "Accuracy per round/epoch"
ordinata = "Accuracy"
plt.figure()

fu.plot_results(rounds_array, accuracy_array_iid, accuracy_array_non_iid, client_nns, titolo, ordinata)
cu.plot_results(rounds_array, accuracy_array_centralized, client_nns, titolo, ordinata)
plt.savefig("accuracy_plot_" + str(client_nns) + ".png")

# Plot della loss
titolo = "Loss per round/epoch"
ordinata = "Loss"
plt.figure()
fu.plot_results(rounds_array, loss_array_iid, loss_array_non_iid, client_nns, titolo, ordinata)
cu.plot_results(rounds_array, loss_array_centralized, client_nns, titolo, ordinata)
plt.savefig("loss_plot_" + str(client_nns) + ".png")



print("Done!")
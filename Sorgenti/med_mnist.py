# Importiamo le librerie necessarie
import torch
from torch import nn

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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
learning_rate_iid = 0.653
learning_rate_non_iid = 0.154
learning_rate_centralized = 0.15
local_epochs = 1

# Inizializzazione della funzione cross-entropy
loss_fn = nn.CrossEntropyLoss()

# Definizione di un ottimizzatore (considero quello del caso iid)
iid_optimizer = torch.optim.SGD(central_model_iid.parameters(), lr=learning_rate_iid)
non_iid_optimizer = torch.optim.SGD(central_model_non_iid.parameters(), lr=learning_rate_non_iid)
centralized_optimizer = torch.optim.SGD(centralized_model.parameters(), lr=learning_rate_centralized)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
rounds = 10
rounds_array = []
accuracy_array_iid = []
accuracy_array_non_iid = []
accuracy_array_centralized = []

for t in range(rounds):
    print(f"Round {t+1}\n-------------------------------")
    fu.fed_train_loop(central_model_iid, iid_loaders, loss_fn, iid_optimizer, client_nns, local_epochs)
    fu.fed_train_loop(central_model_non_iid, non_iid_loaders, loss_fn, non_iid_optimizer, client_nns, local_epochs)
    cu.train_loop(centralized_loader, centralized_model, loss_fn, centralized_optimizer)
    print("---IID---\n")
    fu.validation_loop(validation_dataloader, central_model_iid, loss_fn, accuracy_array_iid)
    print("---NON IID---\n")
    fu.validation_loop(validation_dataloader, central_model_non_iid, loss_fn, accuracy_array_non_iid)
    print("---CENTRALIZZATO---\n")
    cu.validation_loop(validation_dataloader, centralized_model, loss_fn, accuracy_array_centralized)
    rounds_array.append(t+1)

print("Errore sul dataset di test\n")
print("---IID---\n")
fu.test_loop(test_dataloader, central_model_iid, loss_fn, accuracy_array_iid)
print("---NON IID---\n")
fu.test_loop(test_dataloader, central_model_non_iid, loss_fn, accuracy_array_non_iid)
print("---CENTRALIZZATO---\n")
cu.test_loop(test_dataloader, centralized_model, loss_fn, accuracy_array_centralized)

# Plot dei risultati
fu.plot_results(rounds_array, accuracy_array_iid, accuracy_array_non_iid, client_nns)
cu.plot_results(rounds_array, accuracy_array_centralized)

plt.savefig("accuracy_plot_" + str(client_nns) + ".png")
print("Done!")
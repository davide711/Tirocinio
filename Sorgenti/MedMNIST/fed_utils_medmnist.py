import torch
import numpy as np


# Split IID dei dati %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import random

from torch.utils.data import Subset, DataLoader

def split_dataset_iid(training_data, client_nns):

    # Creazione di una lista che contiene, in ogni posizione, 
    # la lista di indici del dataset la cui label corrisponde 
    # alla posizione
    # {label 0: [1, 5, 4, ...], label 1: [2, 7, 15, ...], ...}
    label_to_indices = {}
    for idx, lbl in enumerate(training_data.labels):
    # NOTA: .labels restituisce il dataset sottoforma delle 
    # label di ogni dato
        lbl = int(lbl)
        if lbl not in label_to_indices:
            label_to_indices[lbl] = []
        
        # Aggiungo l'indice alla lista della label corrispondente
        label_to_indices[lbl].append(idx)

    # Array che contiene gli indici del dataset che ogni client riceverà
    clients_indices = [[] for i in range(client_nns)]  # Lista di client_nns liste vuote
    
    # Itero su ogni elemento della lista, ovvero coppie (label, lista di indici), considerando una label per volta
    for lbl, indexes in label_to_indices.items():     
        random.shuffle(indexes)

        # Calcolo il numero di esempi per client, compreso il resto da distribuire ai primi
        total = len(indexes)
        base = total // client_nns
        resto = total % client_nns

        start = 0

        # Ciclo per assegnare un numero pari di indici della label a ogni client
        for client in range(client_nns):
            take = base + (1 if client < resto else 0)
            
            # Aggiungo "take" indici alla liste di indici del client "client", a partire dalla posizione start
            clients_indices[client].extend(indexes[start:start + take])
            
            # Avanza il cursore di "take" posizioni
            start += take

    # Creazione di un DataLoader per ogni client
    dataloaders = []
    for idxs in clients_indices:
        subset = Subset(training_data, idxs)
        loader = DataLoader(subset,
                            batch_size=64,
                            shuffle=True) 
        dataloaders.append(loader)

    return dataloaders

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Split non-IID dei dati %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def split_dataset_non_iid(dataset, n_clients):

    num_labels = len(torch.unique(torch.tensor(dataset.labels).squeeze()))

    # Array di labels casuale
    labels =  np.random.permutation(num_labels)

    # Riempio un array in cui ogni elemento rappresenta le labels per il client
    client_labels = [[] for _ in range(n_clients)]

    
    for i in range(num_labels):
        client_id = i % n_clients
        client_labels[client_id].append(labels[i])

    # Creo una struttura dati contenente gli indici di dati corrispondenti alle label
    labels_indices = {label: [] for label in range(num_labels)}

    for idx, (_, label) in enumerate(dataset):
        labels_indices[int(label.item())].append(idx)

    # Il ciclo itera il dataset (enumerate(array) restituisce sia indice che elemento, quindi una coppia 
    # del tipo (index, (image, label)), dato che il dataset è formato proprio da tuple nella forma (immagine, label))
    # ignorando l'immagine (nel senso di Oggetto Immagine). Riempie quindi la struttura dati con gli indici del dataset
    # corrispondenti alla label individuata dalla posizione:
    #   --> labels_indices[0] contiene gli indici degli elementi del dataset che hanno come label 0

    clients_datasets = []
    for client_id, labels in enumerate(client_labels):
        # Unisci gli indici per le label assegnate a questo client
        indices = []
        for label in labels:
            indices.extend(labels_indices[label])
        # Crea il sotto-dataset
        client_subset = Subset(dataset, indices)
        clients_datasets.append(client_subset)

    # Crea dataloader
    client_loaders = [
        DataLoader(client_dataset, batch_size=64, shuffle=True)
        for client_dataset in clients_datasets
    ]

    return client_loaders

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione della CNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 24, 5)
        self.fc1 = nn.Linear(24 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione del ciclo di training locale %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def client_train_loop(train_dataloader, model, loss_fn, optimizer):
    
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.squeeze()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione dei client %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def client_nn(client_dataloader, model, loss, optim, epochs):

    # Ripetizione del ciclo di training per il numero di epoche
    for i in range(epochs):
        client_train_loop(client_dataloader, model, loss, optim)

    # Restituzione dei dati al server
    return [p.detach().clone() for p in model.parameters()]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Calcolo della media pesata dei pesi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def weights_avg(clients_weights_list, clients_sizes):
    total_data = sum(clients_sizes)
    n_params = len(clients_weights_list[0])

    averaged_weights = []
    for param_index in range(n_params):
        # Somma dei parametri
        weighted_sum = sum(weight[param_index]*n for weight, n in zip(clients_weights_list, clients_sizes)) / total_data

        # media dei parametri
        averaged_weights.append(weighted_sum)
    
    return averaged_weights

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione del ciclo di training federato %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def fed_train_loop(central_model, client_dataloaders, loss_fn, optimizer, client_nns, epochs):
    client_params = [None] * client_nns
    for i in range(client_nns): # Raccogliamo i dati dai client
        # Raccolta dei parametri dai clienti
        client_params[i] = client_nn(client_dataloaders[i], central_model, loss_fn, optimizer, epochs)

    # Media pesata dei parametri
    clients_sizes = [len(loader.dataset) for loader in client_dataloaders]
    updated_params = weights_avg(client_params, clients_sizes)

    # Aggiorno i dati del modello
    with torch.no_grad():
        for param, new_param in zip(central_model.parameters(), updated_params):
            param.data.copy_(new_param)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione del ciclo di validation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def validation_loop(dataloader, model, loss_fn, acc_array):
    model.eval()                                        # Imposta il modello in modalità "valutazione" (disabilita dropout, batchnorm)

    size = len(dataloader.dataset)                 # Numero totale di esempi nel dataset
    num_batches = len(dataloader)                  # Numero di batch nel test set (decisi nel momento della creazione del dataset)
    test_loss, correct = 0, 0                           # Inizializza perdita totale e numero di predizioni corrette

    with torch.no_grad():                               # Disabilita il calcolo dei gradienti che avverrebbe con la loss_fn (qui non serve farlo)
        for X, y in dataloader:                    # Itera sui batch del test set
            pred = model(X)
            y = y.squeeze()                             # Esegue la previsione
            test_loss += loss_fn(pred, y).item()        # Aggiunge la perdita del batch alla somma totale
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Conta i risultati corretti

    test_loss /= num_batches                            # Calcola la media delle perdite
    correct /= size                                     # Calcola l'accuratezza totale
    print(f"Validation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    acc_array.append(100*correct)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Definizione del ciclo di test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def test_loop(test_dataloader, model, loss_fn, acc_array):
    model.eval()                                        # Imposta il modello in modalità "valutazione" (disabilita dropout, batchnorm)

    size = len(test_dataloader.dataset)                 # Numero totale di esempi nel dataset
    num_batches = len(test_dataloader)                  # Numero di batch nel test set (decisi nel momento della creazione del dataset)
    test_loss, correct = 0, 0                           # Inizializza perdita totale e numero di predizioni corrette

    with torch.no_grad():                               # Disabilita il calcolo dei gradienti che avverrebbe con la loss_fn (qui non serve farlo)
        for X, y in test_dataloader:                    # Itera sui batch del test set
            pred = model(X)
            y = y.squeeze()                             # Esegue la previsione
            test_loss += loss_fn(pred, y).item()        # Aggiunge la perdita del batch alla somma totale
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Conta i risultati corretti

    test_loss /= num_batches                            # Calcola la media delle perdite
    correct /= size                                     # Calcola l'accuratezza totale
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Plot dei risultati %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt

def plot_results(rounds_array, accuracy_array_iid, accuracy_array_non_iid, client_nns):
    plt.plot(rounds_array, accuracy_array_iid, marker='o', label='IID')
    plt.plot(rounds_array, accuracy_array_non_iid, marker='s', label='Non-IID')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per round - " + str(client_nns) + " clients")
    plt.legend()
    plt.grid(True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
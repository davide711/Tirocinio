# Importiamo le librerie necessarie
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

# ------------------ 1. DOWNLOAD E PREPARAZIONE DEI DATI -------------------------------------

# Dataset di training (train = True), con immagini trasformate in tensori e etichette one-hot
training_data = datasets.FashionMNIST(
    root="data",            
    train=True,
    download=True,          
    transform=ToTensor(),   
)

# Dataset di test (train = False) (puoi usare lo stesso target_transform o lasciarlo standard)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Definizione del numero di reti client
client_nns = 5

# Suddivisione del dataset in modo IID
def split_dataset_iid(training_data, client_nns):
    
    # Calcolo delle dimensioni delle sottoparti
    total_size = len(training_data)
    part_size = total_size // client_nns
    sizes = [part_size] * client_nns

    # Distribuzione degli elementi rimanenti (se c'è resto)
    for i in range(total_size % client_nns):
        sizes[i] += 1

    # Suddivido randomicamente  il dataset in client_nns parti in base alle sizes calcolate
    train_subsets = random_split(training_data, sizes)

    # Creazione di un DataLoader per ogni subset
    return [DataLoader(subset, batch_size=64, shuffle=True) for subset in train_subsets]


# Suddivisione del dataset in modo NON-IID
def split_dataset_non_iid(dataset, n_clients):

    num_labels = len(torch.unique(dataset.targets))

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
        labels_indices[int(label)].append(idx)

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

# Suddivido il dataset in due modi
iid_loaders = split_dataset_iid(training_data, client_nns)
non_iid_loaders = split_dataset_non_iid(training_data, client_nns)

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# ------------------ 2. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Definizione della rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x                      


# Creazione dei due modelli centrale
central_model_iid = NeuralNetwork().to(device)
central_model_non_iid = NeuralNetwork().to(device)

# ------------------- 3. DEFINIZIONE DEI CLIENT E DEL LORO CICLO DI TRAINING ---------------------

# Definizione del ciclo di training
def client_train_loop(train_dataloader, model, loss_fn, optimizer):
    
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()    



#Definizione del client
def client_nn(client_dataloader, model, loss, optim):
    # Numero di epoche di training del client
    epochs = 1

    # Ripetizione del ciclo di training per il numero di epoche
    for i in range(epochs):
        client_train_loop(client_dataloader, model, loss, optim)

    # Restituzione dei dati al server
    return [p.detach().clone() for p in model.parameters()]

# ---------------------------- 4. PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate_iid = 2e-1
learning_rate_non_iid = 6e-2
batch_size = 64
epochs = 5


# Inizializzazione della funzione cross-entropy
loss_fn = nn.CrossEntropyLoss()


# Definizione di un ottimizzatore (considero quello del caso iid)
iid_optimizer = torch.optim.SGD(central_model_iid.parameters(), lr=learning_rate_iid)
non_iid_optimizer = torch.optim.SGD(central_model_non_iid.parameters(), lr=learning_rate_non_iid)


# Definizione del ciclo di training decentralizzato
# client_params_iid = [None] * client_nns
# client_params_non_iid = [None] * client_nns

# Calcolo della media dei pesi
def weights_avg(clients_weights_list):
    n_clients = len(clients_weights_list)
    n_params = len(clients_weights_list[0])

    averaged_weights = []
    for param_index in range(n_params):
        # Somma dei parametri
        somma = sum(client[param_index] for client in clients_weights_list)

        # media dei parametri
        media = somma / n_clients
        averaged_weights.append(media)
    
    return averaged_weights

# Ciclo di training decentralizzato
def fed_train_loop(central_model, client_dataloaders, loss_fn, optimizer):
    client_params = [None] * client_nns
    for i in range(client_nns): # Raccogliamo i dati dai client
        # Raccolta dei parametri dai clienti
        client_params[i] = client_nn(client_dataloaders[i], central_model, loss_fn, optimizer)

    # Media pesata dei parametri
    updated_params = weights_avg(client_params)

    # Aggiorno i dati del modello
    with torch.no_grad():
        for param, new_param in zip(central_model.parameters(), updated_params):
            param.data.copy_(new_param)



# Definizione del ciclo di test
def test_loop(test_dataloader, model, loss_fn, acc_array):
    model.eval()                                        # Imposta il modello in modalità "valutazione" (disabilita dropout, batchnorm)

    size = len(test_dataloader.dataset)                 # Numero totale di esempi nel dataset
    num_batches = len(test_dataloader)                  # Numero di batch nel test set (decisi nel momento della creazione del dataset)
    test_loss, correct = 0, 0                           # Inizializza perdita totale e numero di predizioni corrette

    with torch.no_grad():                               # Disabilita il calcolo dei gradienti che avverrebbe con la loss_fn (qui non serve farlo)
        for X, y in test_dataloader:                    # Itera sui batch del test set
            pred = model(X)                             # Esegue la previsione
            test_loss += loss_fn(pred, y).item()        # Aggiunge la perdita del batch alla somma totale
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Conta i risultati corretti

    test_loss /= num_batches                            # Calcola la media delle perdite
    correct /= size                                     # Calcola l'accuratezza totale
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    acc_array.append(100*correct)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
rounds = 10

rounds_array = []
accuracy_array_iid = []
accuracy_array_non_iid = []

for t in range(rounds):
    print(f"Round {t+1}\n-------------------------------")
    fed_train_loop(central_model_iid, iid_loaders, loss_fn, iid_optimizer)
    fed_train_loop(central_model_non_iid, non_iid_loaders, loss_fn, non_iid_optimizer)
    print("---IID---\n")
    test_loop(test_dataloader, central_model_iid, loss_fn, accuracy_array_iid)
    print("---NON IID---\n")
    test_loop(test_dataloader, central_model_non_iid, loss_fn, accuracy_array_non_iid)
    rounds_array.append(t+1)

# Plot dei risultati
plt.plot(rounds_array, accuracy_array_iid, marker='o', label='IID')
plt.plot(rounds_array, accuracy_array_non_iid, marker='s', label='Non-IID')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy per round - " + str(client_nns) + " clients")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot_" + str(client_nns) + ".png")


print("Done!")


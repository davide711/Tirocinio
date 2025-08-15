# Test su Fed-Heart-Disease (fhd) con sistema federato


# Importiamo le librerie necessarie
import torch
from torch import nn

import matplotlib.pyplot as plt

# ------------------ 1. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Definizione della rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Sequenza di layer: due hidden layer con ReLU e un output layer da 10 classi
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),          
            nn.Linear(64, 32),   
            nn.ReLU(),              
            nn.Linear(32, 1)      
        )

    def forward(self, x):
        x = self.flatten(x)                
        logits = self.linear_relu_stack(x) 
        return logits                      


# Creazionie del modello centrale
central_model = NeuralNetwork().to(device)

print("First layer weights sum:", central_model.linear_relu_stack[0].weight.sum().item())
# ------------------------
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

# ------------------- 3. DEFINIZIONE DEI CLIENT E DEL LORO CICLO DI TRAINING ---------------------

# Definizione del ciclo di training
def client_train_loop(train_dataloader, model, loss_fn, optimizer):
    # size = len(train_dataloader.dataset)               # Numero totale di esempi nel dataset
    model.train()                                # Imposta il modello in modalità "training" (serve per dropout e batchnorm)

    for batch, (X, y) in enumerate(train_dataloader):  # Itera sui batch: X = immagini, y = etichette (così stampa i risultati ogni 100 batch)
        pred = model(X)
                                 # Esegue la previsione del modello
        loss = loss_fn(pred, y)                  # Calcola la perdita confrontando predizioni e etichette vere

        loss.backward()                          # Calcola il gradiente tramite backpropagation
        optimizer.step()                         # Aggiorna i pesi del modello in base al gradiente
        optimizer.zero_grad()                    # Azzera i gradienti per evitare accumulo nel prossimo step



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
learning_rate = 0.1

# Inizializzazione della funzione cross-entropy
loss_fn = nn.BCEWithLogitsLoss()

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model.parameters(), lr=learning_rate)

# Definizione del ciclo di training decentralizzato
client_params = [None] * client_nns

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
            pred_label = (pred > 0.5).float()
            correct += (pred_label == y).float().sum().item()

    test_loss /= num_batches                            # Calcola la media delle perdite
    print(correct)
    print(size)
    correct /= size                                     # Calcola l'accuratezza totale
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    acc_array.append(100*correct)

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
rounds = 10

rounds_array = []
accuracy_array = []

for t in range(rounds):
    print(f"Round {t+1}\n-------------------------------")
    fed_train_loop(central_model, train_dataloaders, loss_fn, optimizer)
    test_loop(test_dataloaders, central_model, loss_fn, accuracy_array)
    rounds_array.append(t+1)

# Plot dei risultati
plt.plot(rounds_array, accuracy_array, marker='o')
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy per round")
plt.grid(True)
plt.savefig("accuracy_plot.png")


print("Done!")

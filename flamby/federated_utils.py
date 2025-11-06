import torch
from torch import nn

# ***** Definizione della rete neurale *****

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

# ******************************************

# ***** Definizione del ciclo di training dei client *****

def client_train_loop(train_dataloader, model, loss_fn, optimizer):
    
    model.train()                                # Imposta il modello in modalità "training" (serve per dropout e batchnorm)

    for batch, (X, y) in enumerate(train_dataloader):  # Itera sui batch: X = immagini, y = etichette (così stampa i risultati ogni 100 batch)
        pred = model(X)
                                 # Esegue la previsione del modello
        loss = loss_fn(pred, y)                  # Calcola la perdita confrontando predizioni e etichette vere

        loss.backward()                          # Calcola il gradiente tramite backpropagation
        optimizer.step()                         # Aggiorna i pesi del modello in base al gradiente
        optimizer.zero_grad()                    # Azzera i gradienti per evitare accumulo nel prossimo step

# **********************************************************

# ***** Definizione del client *****

def client_nn(client_dataloader, model, loss, optim, client_epochs):

    # Ripetizione del ciclo di training per il numero di epoche
    for i in range(client_epochs):
        client_train_loop(client_dataloader, model, loss, optim)

    # Restituzione dei dati al server
    return [p.detach().clone() for p in model.parameters()]

# ************************************

# ***** Calcolo della media dei pesi *****

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

# ****************************************

# ***** Ciclo di training decentralizzato *****

def fed_train_loop(central_model, client_dataloaders, loss_fn, optimizer, client_nns, client_epochs):
    
    client_params = [None] * client_nns

    for i in range(client_nns): # Raccogliamo i dati dai client
        # Raccolta dei parametri dai clienti
        client_params[i] = client_nn(client_dataloaders[i], central_model, loss_fn, optimizer, client_epochs)

    # Media pesata dei parametri
    updated_params = weights_avg(client_params)

    # Aggiorno i dati del modello
    with torch.no_grad():
        for param, new_param in zip(central_model.parameters(), updated_params):
            param.data.copy_(new_param)

# **********************************************

# ***** Definizione dell'early stopping *****

best_val_loss = float('inf')
counter = 0

def early_stopping(patience, test_loss):
    global counter
    global best_val_loss
    val_loss = test_loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # torch.save(model.state_dict(), 'best_model.pt')  # Salva modello
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping")
            return 1

# *******************************************

# ***** Definizione del ciclo di test *****

def test_loop(test_dataloader, model, loss_fn, acc_array, patience):
    model.eval()                                        # Imposta il modello in modalità "valutazione" (disabilita dropout, batchnorm)

    size = len(test_dataloader.dataset)                 # Numero totale di esempi nel dataset
    num_batches = len(test_dataloader)                  # Numero di batch nel test set (decisi nel momento della creazione del dataset)
    test_loss, correct = 0, 0                           # Inizializza perdita totale e numero di predizioni corrette

    with torch.no_grad():                               # Disabilita il calcolo dei gradienti che avverrebbe con la loss_fn (qui non serve farlo)
        for X, y in test_dataloader:                    # Itera sui batch del test set
            pred = model(X)                             # Esegue la previsione
            test_loss += loss_fn(pred, y).item()        # Aggiunge la perdita del batch alla somma totale
            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred_label == y).float().sum().item()

    test_loss /= num_batches                            # Calcola la media delle perdite
    correct /= size                            # Calcola l'accuratezza totale
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    acc_array.append(100*correct)


    if(early_stopping(patience, test_loss) == 1):
        return 1

# ******************************************

# ***** Plot dei risultati *****

import matplotlib.pyplot as plt

def plot(rounds_array, accuracy_array):
    plt.plot(rounds_array, accuracy_array, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per round")
    plt.grid(True)
    plt.savefig("accuracy_plot.png")

# ********************************
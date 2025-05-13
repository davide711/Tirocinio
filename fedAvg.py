# Importiamo le librerie necessarie
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.data import DataLoader
from torch import nn

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

# Divido il dataset di training in 3 set che passerò ai 3 client
part1, part2, part3 = random_split(training_data, [20000, 20000, 20000])

# Creo un dataloader per ogni client
client_dataloaders = [
    DataLoader(part1, batch_size=64, shuffle=True),
    DataLoader(part2, batch_size=64, shuffle=True),
    DataLoader(part3, batch_size=64, shuffle=True),
]

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# ------------------ 2. CREAZIONE DELLA RETE NEURALE CENTRALE -----------------------------

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
            nn.Linear(28*28, 512),
            nn.ReLU(),          
            nn.Linear(512, 512),   
            nn.ReLU(),              
            nn.Linear(512, 10)      
        )

    def forward(self, x):
        x = self.flatten(x)                
        logits = self.linear_relu_stack(x) 
        return logits                      


# Creazionie del modello centrale
central_model = NeuralNetwork().to(device)

# ------------------- Definizione dei client ---------------------

# Definizione del ciclo di training

def client_train_loop(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)               # Numero totale di esempi nel dataset
    model.train()                                # Imposta il modello in modalità "training" (serve per dropout e batchnorm)

    for batch, (X, y) in enumerate(train_dataloader):  # Itera sui batch: X = immagini, y = etichette (così stampa i risultati ogni 100 batch)
        pred = model(X)                          # Esegue la previsione del modello
        loss = loss_fn(pred, y)                  # Calcola la perdita confrontando predizioni e etichette vere

        loss.backward()                          # Calcola il gradiente tramite backpropagation
        optimizer.step()                         # Aggiorna i pesi del modello in base al gradiente
        optimizer.zero_grad()                    # Azzera i gradienti per evitare accumulo nel prossimo step

        #if batch % 100 == 0:                     # Ogni 100 batch, stampa lo stato della perdita
            #loss, current = loss.item(), batch * batch_size + len(X)
                # - Converte il tensore di perdita in numero float
                # - Aggiorna il numero di esempi visti finora
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Stampa la perdita e il progresso



#Definizione dei 3 client

def nn1(client_dataloader, model, loss, optim):
    client_train_loop(client_dataloader, model, loss, optim)
    return [p.detach().clone() for p in model.parameters()]

def nn2(client_dataloader, model, loss, optim):
    client_train_loop(client_dataloader, model, loss, optim)
    return [p.detach().clone() for p in model.parameters()]

def nn3(client_dataloader, model, loss, optim):
    client_train_loop(client_dataloader, model, loss, optim)
    return [p.detach().clone() for p in model.parameters()]

# -------------------------------------------------------------------------------

# ---------------------------- PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate = 1e-3
batch_size = 64
epochs = 5



# Inizializzazione della funzione cross-entropy
loss_fn = nn.CrossEntropyLoss()



# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(central_model.parameters(), lr=learning_rate)



# Definizione del ciclo di training decentralizzato
client_params = [None] * 3

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
    for i in [0, 10]: # Supponiamo di fare 10 round
        # Raccolta dei parametri dai clienti
        client_params[0] = nn1(client_dataloaders[0], central_model, loss_fn, optimizer)
        client_params[1] = nn2(client_dataloaders[1], central_model, loss_fn, optimizer)
        client_params[2] = nn3(client_dataloaders[2], central_model, loss_fn, optimizer)

        # Media pesata dei parametri
        updated_params = weights_avg(client_params)

        # Aggiorno i dati del modello
        with torch.no_grad():
            for param, new_param in zip(central_model.parameters(), updated_params):
                param.data.copy_(new_param)



# Definizione del ciclo di test
def test_loop(test_dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Ripetiamo train e test loop per 10 epoche, stampando i risultati
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    fed_train_loop(central_model, client_dataloaders, loss_fn, optimizer)
    test_loop(test_dataloader, central_model, loss_fn)
print("Done!")


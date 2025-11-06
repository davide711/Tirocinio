import torch
from torch import nn

# ------------------ 1. DOWNLOAD E PREPARAZIONE DEI DATI -------------------------------------

# Importiamo le librerie necessarie
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import torch

# Trasformazione sulle etichette: da int → one-hot vector (cioè un vettore di tutti 0 tranne l'elemento all'indice corrispondente
# all'immagine che vale 1)
one_hot_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

# Dataset di training (train = True)
training_data = datasets.FashionMNIST(
    root="data",            
    train=True,
    download=True,          
    transform=ToTensor(), 
)

# Dataset di test (train = False)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Dataloader di training
train_dataloader = DataLoader(
    training_data, 
    batch_size=64, 
    shuffle=True
)

#Dataloader di test
test_dataloader = DataLoader(
    test_data, 
    batch_size=64, 
    shuffle=True
)

# I tensori sono strutture dati specializzate simili agli array di NumPy, con la differenza che possono essere eseguiti 
# su GPU o altri acceleratori hardware, che è quello che ci serve per addestrare la rete. In PyTorch, utilizziamo i tensori 
# per codificare gli input e gli output di un modello, nonché i parametri del modello.

from torch.utils.data import DataLoader

# Crea un DataLoader per i dati di addestramento
# - batch_size=64: divide i dati in blocchi da 64 esempi
# - shuffle=True: mescola i dati a ogni epoca

# Crea un DataLoader per i dati di test
# - anche qui usa batch da 64
# - shuffle=True è opzionale (utile solo se vuoi mescolare anche i dati di test)


# Il DataLoader permette di iterare sui dati di training e di test
# --------------------------------------------------------------------------------

# ------------------ 2. CREAZIONE DELLA RETE NEURALE -----------------------------

# Imposta il dispositivo: usa CUDA se disponibile, altrimenti CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# NB: in python posso usare un oggetto come se fosse una funzione, cioè con il suo nome e dei parametri tra parentesi. 
# Questo scatena l'invocazione del metodo __call__ che definisce cosa fare nel caso venga usato come una funzione.
# In questi appunti vedi ad esempio self.flatten o model(x)

# Definizione della rete neurale
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer che appiattisce l'immagine da [1, 28, 28] a [784] e salva l'oggetto layer returnato da nn.Flatten() 
        # in un attributo di classe
        self.flatten = nn.Flatten()
        
        # Sequenza di layer: due hidden layer con ReLU e un output layer da 10 classi
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # Layer fully connected: 784 -> 512
            nn.ReLU(),              # Attivazione non lineare
            nn.Linear(512, 512),    # Altro layer fully connected: 512 -> 512
            nn.ReLU(),              # Altra ReLU
            nn.Linear(512, 10)      # Output layer: 512 -> 10 classi (es. cifre 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)                # Appiattisce l'immagine 
        logits = self.linear_relu_stack(x) # Passa l'input nella rete sequenziale
        return logits                      # Restituisce i logits (non normalizzati)

# Crea il modello e spostalo sul dispositivo (GPU o CPU)
model = NeuralNetwork().to(device)
# -------------------------------------------------------------------------------

# ---------------------------- PROVA DELLA RETE ---------------------------------

# Definizione iperparametri
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Inizializzazione della funzione cross-entropy
loss_fn = nn.CrossEntropyLoss()

# Definizione di un ottimizzatore
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Definizione del ciclo di training
def train_loop(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)               # Numero totale di esempi nel dataset
    model.train()                                # Imposta il modello in modalità "training" (serve per dropout e batchnorm)

    for batch, (X, y) in enumerate(train_dataloader):  # Itera sui batch: X = immagini, y = etichette (così stampa i risultati ogni 100 batch)
        pred = model(X)                          # Esegue la previsione del modello
        loss = loss_fn(pred, y)                  # Calcola la perdita confrontando predizioni e etichette vere

        loss.backward()                          # Calcola il gradiente tramite backpropagation
        optimizer.step()                         # Aggiorna i pesi del modello in base al gradiente
        optimizer.zero_grad()                    # Azzera i gradienti per evitare accumulo nel prossimo step

        if batch % 100 == 0:                     # Ogni 100 batch, stampa lo stato della perdita
            loss, current = loss.item(), batch * batch_size + len(X)
                # - Converte il tensore di perdita in numero float
                # - Aggiorna il numero di esempi visti finora
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Stampa la perdita e il progresso

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
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
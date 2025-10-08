# Federated Learning
Federated Learning (FL)  è un paradigma di apprendimento automatico che consente a più dispositivi o organizzazioni di collaborare all’addestramento di un modello condiviso (ad esempio una rete neurale), 
senza dover centralizzare i dati. Ogni partecipante (client) allena localmente il modello sui propri dati e invia solo aggiornamenti 
(come i nuovi parametri del modello) a un server centrale. Questo approccio permette di preservare la privacy dei dati sensibili e ridurre il traffico di rete.

## Prima simulazione: fashionMNIST
Come prima simulazione consideriamo i client come delle funzioni locali che restituiscono i risultati del proprio ciclo di training sottoforma di 
parametri della rete neurale che stanno allenando. Faremo due esperimenti, entrambi su 10 round di scambio di dati, suddividendo in modo diverso il dataset tra i client:
- Dataset diviso uniformemente tra i client (IID)
- Dataset diviso NON uniformemente tra i client (non-IID)

### Suddivisione IID del Dataset
La suddivisione IID dei dati avviene nel seguente modo:
1. Si creaa una struttura dati che memorizzi, per ogni label, un array di posizioni, ovvero gli indici corrispondenti dei dati nel dataset che hanno quella label. Ogni entry sarà quindi del tipo (label, array-di-indici);
2. Iterando sulle entry, si divide la dimensione dell'array di indici corrispondente per il numero di client, considerando la distribuzione di eventuale resto, ottenendo così la quantità di esempi con quella label che ogni client avrà;
3. Iterando sui client, si distribuiscono gli esempi, ottenendo così una struttura dati che contiene, per ogni client, la lista degli indici del dataset che quel client utilizzerà per il training locale. Ogni entry sarà quindi del tipo (numero-client, array-di-indici);

### Suddivisione non-IID del Dataset
La suddivisione non-IID dei dati avviene nel seguente modo (supponendo di avere n_clients >= n_labels):
1. Si considerano tutte le label del dataset e tutti i client
2. Si itera sui client associando una label a ciascuno
3. Se rimangono delle label non assegnate una volta terminata la prima iterazione sui client, si riparte dal primo client e si continuano ad assegnare le label rimanenti
4. Si ripete il punto 3 fino a quando tutte le label non saranno state assegnate ai client

### Risultati grafici
Di seguito sono mostrati graficamente i risultati ottenuti.
Sono stati utilizzati i seguenti learning rate:
- 5 client (*Figura 1*):
  - Caso IID: learning rate = 2 · 10⁻¹
  - Caso non-IID: learning rate = 6 · 10⁻²
- 10 client (*Figura 2*):
  - Caso IID: learning rate = 7 · 10⁻²
  - Caso non-IID: learning rate = 2 · 10⁻³
 
Notiamo che, nel caso in cui i client siano 10, ciascuno di essi riceve una sola label, dal momento che questi esperimenti fanno riferimento a *fashionMNIST*, un dataset con 10 label. Questo spiega in parte una accuracy nettamente più bassa nel caso di training con 10 client rispetto al caso in cui i client sono 5, dove ognuno riceve 2 label.

<img src="Immagini/fashionMNIST/accuracy_plot_5.png" alt="Accuracy diagram 5 clients" width="400"> </img> <br>
*Figura 1*

<img src="Immagini/fashionMNIST/accuracy_plot_10.png" alt="Accuracy diagram 5 clients" width="400" /> </img> <br>
*Figura 2*

## MedMNIST
*MedMNIST* è una raccolta di dataset medicali pensata per l’addestramento e la valutazione di modelli di deep learning nel campo dell’imaging biomedico. È un insieme di 12 dataset, ognuno rappresentante un tipo diverso di immagine medica e un diverso compito di classificazione. Il dataset preso in considerazione è OrganSMNIST, composto da 17.500 immagini di training e 7.600 di test. Le immagini riguardano 11 organi (tra cui milza, reni, cistifellea e fegato) visti dal piano sagittale, ovvero sezionati lungo un piano orizzontale, e sono state ottenute tramite tecniche radiografiche.
Questo dataset è stato utilizzato per analizzare e confrontare l'andamento dell'allenamento di una rete neurale in 3 casi:
- Allenamento di una rete neurale centralizzata;
- Allenamento federato, con dataset suddiviso IID;
- Allenamento federato, con dataset suddiviso non-IID;

Come per fashionMNIST, essendo OrganSMNIST un dataset unico e pensato per il training centralizzato, è stato necessario suddividerlo manualmente nei due modi IID e non-IID, riutilizzando le funzioni create per fashionMNIST. Questa suddivisione è piuttosto realistica, in quanto spesso all'allenamento di una rete neurale da utilizzare in ambito biomedico partecipano diversi istituti di ricerca o poli ospedalieri, che allenano un proprio modello sulle immagini raccolte per poi partecipare all'aggiornamento di un modello centrale.

### Risultati ottenuti
- Learning rate centralizzato: 0,0025
- Learning rate IID: 0,015
- Learning rate non-IID: 0,003
<img src="Immagini/MedMNIST/accuracy_plot_med.png" alt="Accuracy diagram 5 clients" width="400"> </img> <br>
*Figura 2*

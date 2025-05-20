# Federated Learning
Federated Learning (FL)  è un paradigma di apprendimento automatico che consente a più dispositivi o organizzazioni di collaborare all’addestramento di un modello condiviso (ad esempio una rete neurale), 
senza dover centralizzare i dati. Ogni partecipante (client) allena localmente il modello sui propri dati e invia solo aggiornamenti 
(come i nuovi parametri del modello) a un server centrale. Questo approccio permette di preservare la privacy dei dati sensibili e ridurre il traffico di rete.

## Prima simulazione
Come prima simulazione consideriamo i client come delle funzioni locali che restituiscono i risultati del proprio ciclo di training sottoforma di 
parametri della rete neurale che stanno allenando. Faremo due esperimenti suddividendo in modo diverso il dataset tra i client:
- Dataset diviso uniformemente tra i client
- Dataset diviso NON uniformemente tra i client

### Dataset diciso uniformemente

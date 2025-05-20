# Federated Learning
Federated Learning (FL)  è un paradigma di apprendimento automatico che consente a più dispositivi o organizzazioni di collaborare all’addestramento di un modello condiviso (ad esempio una rete neurale), 
senza dover centralizzare i dati. Ogni partecipante (client) allena localmente il modello sui propri dati e invia solo aggiornamenti 
(come i nuovi parametri del modello) a un server centrale. Questo approccio permette di preservare la privacy dei dati sensibili e ridurre il traffico di rete.

## Prima simulazione
Come prima simulazione consideriamo i client come delle funzioni locali che restituiscono i risultati del proprio ciclo di training sottoforma di 
parametri della rete neurale che stanno allenando. Faremo due esperimenti suddividendo in modo diverso il dataset tra i client:
- Dataset diviso uniformemente tra i client
- Dataset diviso NON uniformemente tra i client

### Dataset diviso uniformemente
- Figura 1: 5 client
- Figura 2: 10 client

![accuracy_plot5](https://github.com/user-attachments/assets/d91a80ea-6e56-43b9-8eb2-19605757ffa6)
*Figura 1.*
![accuracy_plot10](https://github.com/user-attachments/assets/b558703b-88b1-4602-a025-1a1c37edff72)
*Figura 2.*

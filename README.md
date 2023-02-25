# Relazione Progetto IIA 2023
## Studio sull’applicazione di apprendimento per rinforzo alla gestione di un’intersezione semaforizzata.
#### Riccardo Chimisso 866009 - Alberto Ricci 869271
---

<br/>
<br/>

# Sommario
- ## [Obettivo](#obiettivo)
- ## [Reinforcement Learning](#apprendimento-per-rinforzo)
  - ### [Reinforcement Learning](#reinforcement-learning)
  - ### [Q-Learning](#q-learning)
  - ### [Deep Reinforcement Learning](#deep-reinforcement-learning)
  - ### [Deep Q-Learning](#deep-q-learning)
  - ### [Deep Q-Network](#deep-q-network)
- ## [Strumenti](#strumenti)
  - ### [SUMO](#sumo)
  - ### [Sumo-RL](#sumo-rl)
  - ### [Matplotlib](#matplotlib)
  - ### [Stable Baselines 3](#stable-baselines-3)
  - ### [Python, Anaconda e Jupyter Notebook](#python-anaconda-e-jupyter-notebook)
  - ### [Visual Studio Code](#visual-studio-code)
  - ### [GitHub](#github)
- ## [Ambiente](#ambiente)
- ## [Esperimenti e risultati](#esperimenti-e-risultati)
- ## [Conclusione](#conclusione)

<br/>
<br/>

# Obiettivo

Si vuole confrontare per un particolare tipo di intersezione semaforizzata, riferita d’ora in poi con **2WSI** (**2** **W**ay **S**ingle **I**ntersection), uno schema di gestione dei semafori a ciclo fisso con due diversi schemi di gestione controllati da agenti che hanno appreso per rinforzo.
In particolare, verranno quindi confrontati 3 schemi di controllo:  
- Ciclo fisso: le fasi dei semafori sono fisse e si ripetono sempre uguali.  
- Q-Learning: i semafori sono controllati da un agente che ha appreso per rinforzo usando la tecnica del Q-Learning, discussa in dettaglio più avanti.  
- Deep Q-Learning: i semafori sono controllati da un agente che ha appreso per rinforzo usando la tecnica del Deep Q-Learning, discussa in dettaglio più avanti.  

Ciascuno di questi modelli verrà addestrato con una certa situazione di traffico, per poi testare il risultato con la stessa situazione di traffico usata per l’addestramento e su un’altra situazione di traffico che invece non è stata vista durante l’addestramento.  
Questa scelta è motivata dal voler non solo confrontare i modelli tra di loro, ma anche verificare quanto i modelli ad apprendimento riescano a generalizzare, evitando l’overfitting, e quindi adattarsi a diverse situazioni di traffico.  
La robustezza degli agenti è molto importante in quanto nella realtà è facile che un’intersezione semaforizzata sia soggetta a traffico variabile, basta pensare alla differenza tra orario di punta e notte, oppure mesi feriali e festivi.

<br/>
<br/>

# Apprendimento per rinforzo

## Reinforcement Learning
Il Reinforcement Learning è una tecnica di apprendimento che prevede l’apprendimento di un agente attraverso l’interazione con un ambiente dinamico. L'agente interagisce con l'ambiente in modo sequenziale, compiendo azioni e ricevendo una ricompensa (reward).  
Lo scopo dell'agente è quello di massimizzare la ricompensa cumulativa che viene fornita dall'ambiente in risposta alle sue azioni.  
L'RL si basa su un processo di apprendimento per esplorazione e sperimentazione, in cui l'agente deve scegliere le azioni da effettuare in modo da massimizzare la sua ricompensa. L'agente apprende dalla sua esperienza accumulando conoscenze e sviluppando strategie sempre più efficaci.  
Lo scopo di un agente è quindi *max=Σ(s<sub>t</sub>, a<sub>t</sub>)* per *t = 0* a *T*, dove *T* è il massimo di istanti temporali.  
È da notare che un agente con tale scopo potrebbe trovarsi in indecisione nel caso di sequenze di azioni la cui ricompensa totale sia pari. Ad esempio date le sequenze di ricompensa ⧼ *0, 0, 1* ⧽ e ⧼ *1, 0, 0* ⧽ quale dovrebbe scegliere l’agente? Per decidere si introduce il discount factor *γ* per diminuire il peso che le ricompense future hanno rispetto alle più immediate, così che l’agente scelga la massimizzazione più veloce della ricompensa cumulativa. Il discount factor è *0 ≤ γ ≤ 1* e la ricompensa al tempo *t* è data da:  
*R<sub>t</sub> = r<sub>t</sub> + γr<sub>t+1</sub> + γ<sup>2</sup>r<sub>t+2</sub> + ... + γ<sup>T-t</sup>r<sub>T</sub> = Σγ<sup>i-t</sup>r<sub>i</sub> = r<sub>t</sub> + R<sub>t+1</sub>* per *i = t* a *T*, dove *r<sub>i</sub>* è la ricompensa per la transizione dell’istante di tempo *i*-esimo. Questa sommatoria altro non è che la serie geometrica, e in quanto tale converge sempre a un valore finito anche per *T = ∞*.

## Q-Learning
Il Q-Learning è uno specifico algoritmo di Reinforcement Learning che si basa sulla costruzione di una tabella *Q* che indichi il valore di ricompensa per ogni possibile stato al compimento di una qualsiasi delle azioni possibili.  
Per costruire tale tabella si utilizza una procedura iterativa in cui l'agente esplora l'ambiente eseguendo delle azioni più o meno casuali. In dettaglio, ad ogni passo, la tabella sarà aggiornata con: *Q[s][a] = Q[s][a] + ⍺(r + γ**ᐧ**max(Q[s']) - Q[s][a])*, dove *s* è lo stato attuale, *a* l’azione compiuta, *r* la ricompensa ottenuta, *s'* lo stato successivo, *γ* il discount factor, *⍺* il learning rate e *max(Q[x])* restituisce la massima ricompensa ottenibile dallo stato *x*.  
In questo modo la cella della tabella che rappresenta il valore atteso della ricompensa per il compimento dell'azione a nello stato s convergerà gradualmente all’effettivo valore.  
Come anticipato, la scelta dell’azione da compiere sarà inizialmente casuale, finché non si decide che si è esplorato abbastanza. La politica spesso più usata a questo fine è la *ε*-greedy, dove data una *ε* iniziale che rappresenta la probabilità di compiere un’azione casuale, si diminuisce tale valore al progredire delle iterazioni fino a un minimo.  
Il QL è una tecnica molto potente ed efficace, in grado di apprendere strategie di azione ottimali in una vasta gamma di applicazioni. Tuttavia, può essere sensibile al rumore, all'indeterminazione delle azioni e all’applicazione su ambienti continui. Inoltre, il QL richiede un grande quantità di memoria per memorizzare la tabella *Q*, specialmente quando l'ambiente ha uno spazio di stati (*S*) e di azioni (*A*) possibili (*ϴ(SA)*).

## Deep Reinforcement Learning
Il Deep Reinforcement Learning è una tecnica di apprendimento automatico basata sull’RL, ma che si pone come obiettivo di sopperire alla problematica di quest’ultimo per spazi di stati e azioni molto grandi. Per farlo si utilizzano delle reti neurali profonde (Deep Neural Networks, da cui il nome) per  approssimare i valori della *Q* table senza richiederne le stesse risorse in termini di memoria.

## Deep Q-Learning
L’approccio più semplice per l’implementazione di una rete neurale profonda come approssimatore per la *Q* table consiste nell'utilizzare una rete neurale profonda ad ogni passo per ottenere la ricompensa attesa e aggiornare i pesi della rete neurale tramite il metodo del gradient descent rispetto alla ricompensa effettivamente ottenuta.  
Questo approccio ha però lo svantaggio di non rispettare due condizioni importanti per la probabile convergenza di un metodo di apprendimento supervisionato come le reti neurali:  
- Gli obiettivi (target) della rete neurale non sono stazionari, ovvero variano nel tempo, poiché è la rete stessa che ottiene i target attuali in base alle predizioni che essa stessa fa per i target futuri. Infatti la rete neurale stima i valori di *Q*, che rappresentano l'atteso guadagno futuro associato ad una coppia stato-azione, e questi valori vengono utilizzati per calcolare i target per l'aggiornamento dei pesi della rete stessa. Poiché la rete neurale si aggiorna utilizzando i suoi stessi output per stimare i target futuri, i target non sono fissi e variano continuamente nel tempo e questo rende la rete neurale instabile e prona alla non convergenza.  
- Gli input alla rete neurale non sono indipendenti ed identicamente distribuiti poiché generati da una sequenza temporale con correlazione sequenziale e dipendono dalla *Q* table utilizzata dall'agente, che può cambiare nel tempo a seguito dell'esperienza maturata.

## Deep Q-Network
L’approccio che prende il nome di Deep Q-Network cerca di arginare le problematiche del più semplice DQL tramite i seguenti due metodi:  
- Per diminuire la non stazionarietà dei target si introduce una seconda rete neurale profonda, detta target network, che viene usata per stimare i target a cui deve convergere la rete principale, detta main network, in fase di addestramento. I pesi della target network vengono anch’essi aggiornati con il progredire dell’addestramento, ma con una frequenza molto minore rispetto a quella della main network. In questo modo, è possibile dividere l’addestramento in tanti piccoli problemi di apprendimento supervisionato che vengono presentati all’agente in maniera sequenziale. Questo non solo consente di aumentare la probabilità di convergenza, ma anche migliorare la stabilità del training, sebbene a costo di una velocità minore, in quanto non vengono utilizzati i valori più aggiornati dei target.  
- Per ridurre l’impatto della correlazione tra gli input viene adottata la tecnica Experience Replay, ovvero l’utilizzo di una struttura dati chiamata replay buffer all’interno della quale salvare dei campioni *(s, a, r, s')* raccolti dall’agente durante l’apprendimento così da poterlo addestrare anche su dei gruppi di campioni selezionati casualmente dal replay buffer, che in questo modo permette di rendere gli input un po’ più i.i.d. di quanto effettivamente siano. Inoltre questa tecnica consente di imparare di più dai singoli episodi, richiamare eventi rari e, in generale, fare un uso migliore dell’esperienza accumulata dall’agente.

<br/>
<br/>

# Strumenti
## SUMO
[SUMO](https://sumo.dlr.de/docs/) (Simulation of Urban MObility) è un simulatore open source di mobilità urbana.  
SUMO consente agli sviluppatori di simulare il traffico veicolare e pedonale in un ambiente urbano, consentendo loro di testare e valutare soluzioni di mobilità come semafori intelligenti, veicoli autonomi, car pooling e molto altro ancora.  
Il simulatore è altamente personalizzabile e consente agli utenti di definire le caratteristiche dei veicoli, delle strade e degli incroci, nonché delle condizioni meteorologiche e del traffico, per creare scenari realistici. Inoltre, SUMO offre diverse metriche di valutazione, come il tempo di percorrenza, il consumo di carburante, le emissioni di gas a effetto serra e il tempo di attesa di ciascun veicolo, che possono essere utilizzate per valutare le prestazioni dei sistemi di mobilità.  
Viene utilizzato in questo progetto proprio come simulatore dell’ambiente, creata la rete stradale come 2WSI e le diverse situazioni di traffico da dover gestire.  
SUMO fornisce anche un’API chiamata [TraCI](https://sumo.dlr.de/docs/TraCI.html) (Traffic Control Interface), un'interfaccia di controllo del traffico per consentire l'interazione tra il simulatore di traffico e gli agenti esterni, come ad esempio i sistemi di controllo del traffico intelligenti.  
L'interfaccia messa a disposizione da TraCI è basata su socket che consente agli utenti di controllare i veicoli nel simulatore, modificare le caratteristiche della rete stradale e dei semafori, e ottenere informazioni sullo stato del traffico in tempo reale. Inoltre, TraCI consente anche la registrazione e la riproduzione di scenari di traffico per analizzare i risultati della simulazione. TraCI è supportato da una vasta gamma di linguaggi di programmazione, tra cui Python utilizzato in questo progetto.

## Sumo-RL
[Sumo-RL](https://github.com/LucasAlegre/sumo-rl) è un progetto open source basato su SUMO e TraCI per l’applicazione di algoritmi di RL ad ambienti di simulazione del traffico per la gestione di intersezioni semaforizzate. Fornisce un’interfaccia in Python semplice da utilizzare per la creazione di ambienti in SUMO e la loro gestione tramite algoritmi di RL. In particolare è possibile utilizzare un’implementazione già fatta del QL, inoltre è facilmente possibile integrare gli ambienti forniti con altre implementazioni di algoritmi di altre librerie, come ad esempio la DQN di Stabe Baselines 3, purché tali implementazioni accettino un ambiente Gymnasium (in caso di singolo semaforo) o Petting Zoo (in caso di multipli semafori).  

## Matplotlib
[Matplotlib](https://matplotlib.org/) è una libreria in Python per la creazione di visualizzazioni statiche, animate o interattive. Poiché semplice da usare e già integrata con Jupyter Notebook, viene usata in questo progetto per creare e salvare grafici di varie metriche raccolte durante l’esecuzione degli algoritmi di RL.  
Inizialmente si era anche pensato di permettere la visualizzazione in tempo reale della costruzione dei grafici delle metriche, ma, nonostante si fosse ottenuto tale risultato, si è scelto di escludere questa funzione poiché non valeva l’enorme rallentamento che ricadeva sulle simulazioni.

## Stable Baselines 3
[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) è un’altra libreria open source in Python che fornisce implementazioni di molteplici algoritmi di RL e DRL, di cui in questo progetto si è selezionata quella per DQN, e l’integrazione con ambienti di Gymnasium e Petting Zoo. È stato in realtà necessario installare una versione particolare di SB3 per garantire l’effettiva integrazione con ambienti Gymnasium.

## Python, Anaconda e Jupyter Notebook
Python è stato il linguaggio scelto per questo progetto vista la sua grande adeguatezza nelle applicazioni di apprendimento automatico, e per via del fatto che sia Sumo-RL che Matplotlib sono scritti e facilmente integrabili in Python. Inoltre questo ha permesso l’utilizzo di Jupyter Notebook, che facilita l’esecuzione e la condivisione dello script principale del progetto, permettendo inoltre la visualizzazione dei grafici in real time (come detto però, questo è stato tolto) e la visualizzazione dei grafici completati (anche questa funzione in realtà rimossa per evitare output troppo lunghi, ma è facilmente riattivabile).  
La versione di Python utilizzata è la 3.9.13 tramite [Anaconda](https://www.anaconda.com/products/distribution), poiché SB3 utilizza [Pytorch](https://pytorch.org/) il quale a sua volta necessita di una versione non superiore alla 3.9 di Python. Inoltre tramite Anaconda e Pytorch è stato possibile eseguire le reti neurali direttamente sulle GPU NVIDIA a nostra disposizione.

## Visual Studio Code
[Visual Studio Code](https://code.visualstudio.com/) è stato l’IDE scelto per lo sviluppo del codice del progetto, dai file xml per la realizzazione degli ambienti SUMO, ai file Python e Jupyter Notebook per la scrittura, implementazione e esecuzione delle sperimentazioni. La scelta è ricaduta su questo editor poiché già conosciuto e con una facilissima integrazione di Git, Jupyter Notebook, Python e GitHub.

## GitHub
Infine, [GitHub](https://github.com/) è stato l’ultimo degli strumenti principali utilizzati nella realizzazione di questo progetto, permettendo una facile coordinazione tra sviluppi paralleli (questo più Git che GitHub) e uno spazio per salvare, pubblicare e condividere la repository del progetto, mantenendo comunque l’esclusiva proprietà della stessa. Inoltre ha reso facile la scelta di una licenza per la repository, nonché ha offerto la possibilità di visualizzare propriamente il README in Markdown con in aggiunta istruzioni e spiegazioni più inerenti e dettagliate dell’impostazione, spiegazione e realizzazione del codice del progetto.  
Sebbene il README contenga e integri la relazione stessa, è stata comunque caricata la relazione in PDF nella repository per permettere una più facile visualizzazione della stessa, con il vantaggio di visualizzare meglio l’indice, la divisione tra pagine (e quindi argomenti) e le formule Latex.  
Inoltre la differenza principale tra la relazione e il README è che quest’ultimo è scritto in inglese.

<br/>
<br/>

# Setup


<br/>
<br/>

# Codebase


<br/>
<br/>

# Ambiente

<br/>
<br/>

# Esperimenti e risultati

<br/>
<br/>

# Conclusione

<br/>
<br/>

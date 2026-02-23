Ciao, questo è il mio Project Work per la tesi di laurea in informatica L31

 "Il Feedback dei clienti come dato prezioso tramite il Machine Learning"

Questo progetto nasce dall'esigenza di automatizzare il processo di gestione dei feedback all'interno del settore alberghiero, ma è applicabile a qualsiasi settore e azienda.
L'idea alla base è trasformare il testo non strutturato delle recensioni in dati preziosi per l'azienda.
 Utilizzando algoritmi di Machine Learning, ho sviluppato un sistema capace di analizzare il sentiment dei commenti e di smistarli automaticamente verso il reparto di competenza (Housekeeping, Reception o Food & Beverage). Questo permette di ottimizzare i flussi di lavoro, riducendo i tempi di risposta manuale e garantendo che le criticità vengano segnalate immediatamente ai responsabili corretti.

Il cuore tecnologico del progetto si basa su una pipeline di Natural Language Processing (NLP) che utilizza la vettorizzazione TF-IDF e modelli di Regressione Logistica, scelti per il loro eccellente equilibrio tra prestazioni computazionali e interpretabilità dei risultati. L'intero ecosistema è stato realizzato con strumenti Open Source e un'interfaccia grafica moderna per rendere la tecnologia accessibile anche a personale non tecnico.

Istruzioni per l'utilizzo :


Per avviare correttamente il sistema sul tuo computer, segui questi passaggi in ordine cronologico:

1- Preparazione dell'ambiente:
Assicurati di avere Python installato. Apri il terminale nella cartella del progetto e installa tutte le librerie necessarie lanciando:
pip install -r requirements.txt

2- Generazione dei dati (datagenerator.py):
Esegui per primo questo script. Verrà creato un file chiamato hotel_reviews.csv contenente 600 recensioni sintetiche bilanciate tra i vari reparti e sentiment, complete di termini ambigui per testare la precisione dei modelli.
python datagenerator.py

3- Addestramento della Pipeline ML (ml_pipeline.py):
Una volta pronti i dati, avvia lo script della pipeline. Il sistema pulirà il testo, addestrerà i modelli tramite GridSearch per trovare i parametri migliori e salverà i file .pkl (i modelli "istruiti") nella tua cartella. Vedrai a schermo le metriche di Accuracy e F1-Score.
python ml_pipeline.py

4- Avvio della Dashboard (Dashboard.py):
Infine, lancia l'interfaccia utente con Streamlit. Si aprirà automaticamente una pagina nel tuo browser dove potrai testare il modello inserendo recensioni a mano o caricando file CSV per analisi massive.
streamlit run Dashboard.py



Qui i comandi da eseguire in ordine cronologico nel terminale:

 pip install -r requirements.txt
 python datagenerator.py
 python ml_pipeline.py
 python -m streamlit run Dashboard.py
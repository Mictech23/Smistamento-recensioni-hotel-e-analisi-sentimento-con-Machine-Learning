# Smistamento-recensioni-hotel-e-analisi-sentimento-con-Machine-Learning
Il Feedback dei clienti come dato prezioso tramite il Machine Learning.

Descrizione del Progetto
Questo progetto nasce come elaborato per il corso di laurea in Informatica (L-31). L'obiettivo è trasformare i feedback non strutturati dei clienti alberghieri in dati azionabili per il management.

Il sistema utilizza tecniche di Natural Language Processing (NLP) per analizzare il testo delle recensioni, identificare automaticamente il reparto di competenza (Housekeeping, Reception, Food & Beverage) e valutarne il sentiment (Positivo, Negativo, Neutro).

 Funzionalità Principali:

 
Generazione Dati Sintetici: Un motore custom (Datagenerator.py) che combina template semantici e AI generativa per creare dataset bilanciati.

Pipeline di Machine Learning: Implementazione di una pipeline basata su TF-IDF Vectorization e Logistic Regression, ottimizzata tramite GridSearchCV.

Dashboard Interattiva: Interfaccia sviluppata con Streamlit per l'analisi in tempo reale (Live Analysis) e l'elaborazione massiva di file CSV (Batch Processing).

Business Intelligence: Visualizzazione dinamica dei risultati tramite grafici interattivi (Plotly) per identificare rapidamente le aree critiche aziendali.

Tecnologie Utilizzate:

Linguaggio: Python 3.x

Librerie ML: Scikit-Learn, Pandas, NumPy

Interfaccia & Grafica: Streamlit, Plotly

AI Tool: Google Gemini (per l'arricchimento del dataset sintetico)

Struttura del Repository


Dashboard.py: Il file principale che lancia l'interfaccia utente.

pipeline.py: Script per l'addestramento e la validazione del modello.

Datagenerator.py: Script per la creazione del dataset recensioni_hotel.csv.

requirements.txt: Elenco delle dipendenze necessarie.

models/: Cartella contenente i modelli salvati in formato .pkl (opzionale).

Installazione e Utilizzo:


Clona il repository:

Bash
git clone https://github.com/tuo-username/hotel-feedback-intelligence.git
cd hotel-feedback-intelligence
Installa le dipendenze:

Bash
pip install -r requirements.txt
Lancia l'applicazione:

Bash
python -m streamlit run Dashboard.py

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_models():
    # 1. Caricamento Dati
    if not os.path.exists('hotel_reviews.csv'):
        print("❌ Errore: Esegui prima data_generator.py!")
        return

    df = pd.read_csv('hotel_reviews.csv')
    print("Dati caricati. Inizio addestramento...")
    
    # 2. Preprocessing & Vettorizzazione
    # Usiamo n-grams (1,2) per catturare coppie di parole come "non pulito"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    X = vectorizer.fit_transform(df['review'])
    
    # 3. Split 80/20
    y_dept = df['department']
    y_sent = df['sentiment']
    
    # Creiamo split identici per entrambi i target
    X_train, X_test, y_d_train, y_d_test, y_s_train, y_s_test = train_test_split(
        X, y_dept, y_sent, test_size=0.2, random_state=42
    )

    # 4. Addestramento Modelli (Logistic Regression)
    # Modello A: Reparto
    model_dept = LogisticRegression(max_iter=1000)
    model_dept.fit(X_train, y_d_train)
    
    # Modello B: Sentiment
    model_sent = LogisticRegression(max_iter=1000)
    model_sent.fit(X_train, y_s_train)

    # 5. Valutazione (Opzionale: stampa report)
    print("\n--- Report Reparto ---")
    print(classification_report(y_d_test, model_dept.predict(X_test)))
    print("\n--- Report Sentiment ---")
    print(classification_report(y_s_test, model_sent.predict(X_test)))

    # 6. Salvataggio (Pickling)
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/model_dept.pkl', 'wb') as f:
        pickle.dump(model_dept, f)
    with open('models/model_sent.pkl', 'wb') as f:
        pickle.dump(model_sent, f)

    print("✅ Modelli salvati correttamente nella cartella 'models/'")

if __name__ == "__main__":
    train_models()
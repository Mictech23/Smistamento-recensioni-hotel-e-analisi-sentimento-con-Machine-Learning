import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def preprocess_text(df: pd.DataFrame) -> pd.Series:
    """
    Esegue il preprocessing del testo: unione di titolo e corpo, 
    conversione in minuscolo e rimozione della punteggiatura.
    """
    # Unione dei campi titolo e corpo del testo 
    texts = (df["title"].fillna("") + " " + df["body"].fillna(""))
    # Conversione in minuscolo [cite: 175]
    texts = texts.str.lower()
    # Rimozione punteggiatura [cite: 175]
    texts = texts.str.translate(str.maketrans("", "", string.punctuation))
    return texts

def main(csv_path="hotel_reviews.csv"):
    # Caricamento del dataset generato 
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Errore: file {csv_path} non trovato. Esegui prima datagenerator.py.")
        return

    df = df.dropna(subset=["title", "body", "department", "sentiment"])

    # Preprocessing e definizione target
    X = preprocess_text(df)
    y_dept = df["department"]
    # Mapping binario per il sentiment: 0 = Negativo, 1 = Positivo 
    y_sent = df["sentiment"].map({"Positivo": 1, "Negativo": 0})

    # Definizione stop-wordse
    italian_stop = [
        "il","lo","la","e","a","che","in","un","una","mi","ti","si",
        "per","con","su","del","dei","di","da","al","gli","le","perché",
        "ma","anche","tra","fra","se","sul","sulla","della","delle","degli"
    ]
    # Manteniamo 'non' per non perdere il senso delle negazioni nei bigrammi 
    if "non" in italian_stop:
        italian_stop.remove("non")

    # Vettorizzazione TF-IDF con unigrammi e bigrammi
    vect = TfidfVectorizer(ngram_range=(1, 2), stop_words=italian_stop)
    X_vec = vect.fit_transform(X)

    # Split 80% training e 20% test con stratificazione
    X_train, X_test, yd_train, yd_test, ys_train, ys_test = train_test_split(
        X_vec, y_dept, y_sent, test_size=0.2, random_state=42, stratify=y_dept
    )

    # Configurazione GridSearch per ottimizzare il parametro di regolarizzazione 'C' 
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs']
    }

    print("Inizio addestramento con GridSearch (5-fold CV)...")

    # 1. Modello per il Reparto (Multiclasse: Housekeeping, Reception, F&B) 
    grid_dept = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
    grid_dept.fit(X_train, yd_train)
    model_dept = grid_dept.best_estimator_
    pred_dept = model_dept.predict(X_test)

    # 2. Modello per il Sentiment (Binario: Positivo/Negativo)
    grid_sent = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
    grid_sent.fit(X_train, ys_train)
    model_sent = grid_sent.best_estimator_
    pred_sent = model_sent.predict(X_test)

    # --- Valutazione Modello Reparto ---
    print("\n" + "="*30)
    print("REPORT VALUTAZIONE: REPARTO")
    print("="*30)
    print(f"Migliori parametri: {grid_dept.best_params_}")
    print(f"Accuracy: {accuracy_score(yd_test, pred_dept):.4f}")  # [cite: 210]
    print(f"F1-Score (macro): {f1_score(yd_test, pred_dept, average='macro'):.4f}")  # [cite: 211]
    print("\nMatrice di Confusione:\n", confusion_matrix(yd_test, pred_dept))  # [cite: 238]
    print("\nClassification Report:\n", classification_report(yd_test, pred_dept))  # [cite: 212]

    # --- Valutazione Modello Sentiment ---
    print("\n" + "="*30)
    print("REPORT VALUTAZIONE: SENTIMENT")
    print("="*30)
    print(f"Migliori parametri: {grid_sent.best_params_}")
    print(f"Accuracy: {accuracy_score(ys_test, pred_sent):.4f}")  # [cite: 244]
    print(f"F1-Score (macro): {f1_score(ys_test, pred_sent, average='macro'):.4f}")  # [cite: 245]
    print("\nMatrice di Confusione:\n", confusion_matrix(ys_test, pred_sent))
    print("\nClassification Report:\n", classification_report(ys_test, pred_sent, target_names=["Negativo", "Positivo"]))  # [cite: 242]

    # Salvataggio dei modelli e del vettorizzatore 
    joblib.dump(vect, "vectorizer.pkl")
    joblib.dump(model_dept, "model_department.pkl")
    joblib.dump(model_sent, "model_sentiment.pkl")
    
    print("\n" + "-"*30)
    print("Pipeline completata. Modelli salvati correttamente.")
    print("-"*30)

if __name__ == "__main__":

    main()

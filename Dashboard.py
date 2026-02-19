import streamlit as st
import pickle
import os

# Configurazione Pagina
st.set_page_config(page_title="Piattaforma di gestione dei Feedback dei clienti", page_icon="🏨")

# Funzione per caricare i modelli
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
        vec = pickle.load(f)
    with open(os.path.join(models_dir, 'model_dept.pkl'), 'rb') as f:
        mod_dept = pickle.load(f)
    with open(os.path.join(models_dir, 'model_sent.pkl'), 'rb') as f:
        mod_sent = pickle.load(f)
    return vec, mod_dept, mod_sent

# Header
st.title(" Piattaforma di gestione dei Feedback dei clienti")
st.markdown("""
Questo prototipo analizza le recensioni dei clienti utilizzando il **Machine Learning**.
Identifica automaticamente il **Reparto Competente** e il **Sentiment** del testo.
""")
st.divider()

# Caricamento risorse
try:
    vectorizer, model_dept, model_sent = load_models()
    st.success("Sistemi AI caricati e pronti.", icon="✅")
except Exception as e:
    st.error(f"Errore nel caricamento dei modelli: {e}. Esegui prima ml_pipeline.py!")
    st.stop()

# Input Utente
review = st.text_area("✍️ Incolla qui la recensione del cliente:", height=100, placeholder="Es: La camera era sporca ma la colazione ottima...")

if st.button("🔍 Analizza Recensione", type="primary"):
    if review:
        # Preprocessing e Predizione
        vec_review = vectorizer.transform([review])
        pred_dept = model_dept.predict(vec_review)[0]
        pred_sent = model_sent.predict(vec_review)[0]
        
        # Calcolo Probabilità (Confidenza)
        prob_dept = max(model_dept.predict_proba(vec_review)[0]) * 100
        prob_sent = max(model_sent.predict_proba(vec_review)[0]) * 100

        st.divider()
        st.subheader("📊 Risultati dell'Analisi")

        # Layout a colonne per i risultati
        col1, col2 = st.columns(2)

        with col1:
            st.info("**Reparto Assegnato**")
            st.markdown(f"### 🏢 {pred_dept}")
            st.caption(f"Confidenza modello: {prob_dept:.1f}%")

        with col2:
            st.info("**Sentiment Rilevato**")
            # Logica colori per Sentiment
            if pred_sent == "Positivo":
                st.success(f"### 😊 {pred_sent}")
            elif pred_sent == "Negativo":
                st.error(f"### 😡 {pred_sent}")
            else:
                st.warning(f"### 😐 {pred_sent}")
            st.caption(f"Confidenza modello: {prob_sent:.1f}%")

    else:
        st.warning("Per favore inserisci del testo prima di analizzare.")

# Footer
st.markdown("---")
st.markdown("*Sviuppato da Michele Siddi*")
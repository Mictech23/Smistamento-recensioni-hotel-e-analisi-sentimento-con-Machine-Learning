import streamlit as st
import pandas as pd
import string
import joblib
import plotly.express as px
from datetime import datetime


st.set_page_config(
    page_title="Analisi Recensioni",
    page_icon="📊",
    layout="wide"
)

# custom CSS for a more modern look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size:2.5rem !important;
        font-weight: 600;
        color: #333333;
    }
    .subheader {
        color: #444444;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        padding: 0.4rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


vect = joblib.load("vectorizer.pkl")
model_dept = joblib.load("model_department.pkl")
model_sent = joblib.load("model_sentiment.pkl")


def preprocess_texts(series: pd.Series) -> pd.Series:
    texts = series.fillna("").str.lower()
    texts = texts.str.translate(str.maketrans("", "", string.punctuation))
    return texts


def _display_probabilities(probs, classes, title=""):
    """Draw a horizontal bar chart of probabilities."""
    df = pd.DataFrame({"classe": classes, "prob": probs})
    df = df.sort_values("prob", ascending=True)
    fig = px.bar(
        df,
        x="prob",
        y="classe",
        orientation="h",
        text="prob",
        labels={"prob": "Probabilità", "classe": "Classe"},
        range_x=[0, 1],
        title=title,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250)
    fig.update_traces(texttemplate="%{text:.1%}")
    st.plotly_chart(fig, use_container_width=True)


def predict_single(text: str):
    processed = preprocess_texts(pd.Series([text]))
    vec = vect.transform(processed)
    dept = model_dept.predict(vec)[0]
    dept_probs = model_dept.predict_proba(vec)[0]
    sent = model_sent.predict(vec)[0]
    sent_probs = model_sent.predict_proba(vec)[0]
    
    sent_label = "Positivo" if sent == 1 else "Negativo"
    return {
        "department": dept,
        "department_probs": dept_probs,
        "sentiment": sent_label,
        "sentiment_probs": sent_probs,
    }


def main():
 
    with st.sidebar:
        st.header("Guida")
        st.write("Scegli una delle schede in alto per effettuare un'analisi singola o di massa.")
        st.write("I risultati vengono mostrati con grafici interattivi e puoi esportare il CSV.")
        st.write("Created by Michele Siddi")

    st.title("Dashboard Analisi Recensioni Alberghiere")
    tab1, tab2 = st.tabs(["Analisi Singola", "Analisi Batch"])

    with tab1:
        st.header("Inserisci recensione manualmente")
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                review = st.text_area("Recensione (titolo + testo)")
                if st.button("Analizza"):  # quando viene premuto
                    if not review.strip():
                        st.warning("Inserisci del testo prima di analizzare.")
                    else:
                        res = predict_single(review)
                        # output section will be updated below
                        st.session_state["last_res"] = res
                        st.balloons()
            with col2:
                st.write("*Risultati 👇")
                if "last_res" in st.session_state:
                    r = st.session_state["last_res"]
                    st.metric("Reparto consigliato", r["department"])
                    icon = "🙂" if r["sentiment"] == "Positivo" else "☹️"
                    st.metric("Sentiment", f"{r['sentiment']} {icon}")
       
        if "last_res" in st.session_state:
            r = st.session_state["last_res"]
            st.subheader("Probabilità predizioni")
            _display_probabilities(r["department_probs"], model_dept.classes_, title="Probabilità Reparto")
            _display_probabilities(r["sentiment_probs"], ["Negativo", "Positivo"], title="Probabilità Sentiment")

    with tab2:
        st.header("Analisi Batch da CSV")
        uploaded = st.file_uploader("Carica un file CSV", type="csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            
            if "title" in batch_df.columns and "body" in batch_df.columns:
                text_series = (batch_df["title"].fillna("") + " " + batch_df["body"].fillna(""))
            elif "text" in batch_df.columns:
                text_series = batch_df["text"].astype(str)
            else:
                st.error("Il CSV deve contenere colonne 'title' e 'body' o 'text'.")
                return

            processed = preprocess_texts(text_series)
            X_vec = vect.transform(processed)
            dept_preds = model_dept.predict(X_vec)
            dept_probs = model_dept.predict_proba(X_vec)
            sent_preds = model_sent.predict(X_vec)
            sent_probs = model_sent.predict_proba(X_vec)

            out = batch_df.copy()
            out["pred_department"] = dept_preds
            out["pred_sentiment"] = ["Positivo" if s == 1 else "Negativo" for s in sent_preds]
            for idx, cls in enumerate(model_dept.classes_):
                out[f"prob_dept_{cls}"] = dept_probs[:, idx]
            out["prob_sent_neg"] = sent_probs[:, 0]
            out["prob_sent_pos"] = sent_probs[:, 1]

            st.subheader("Anteprima risultati")
            st.dataframe(out.head())

      
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Totale righe", len(out))
            with col2:
                counts = out["pred_department"].value_counts()
                top = counts.idxmax()
                st.metric("Reparto più frequente", top, counts.max())
            with col3:
                sent_counts = out["pred_sentiment"].value_counts()
                st.metric("Sentiment positivo", f"{sent_counts.get('Positivo',0)}", 
                          delta=f"{sent_counts.get('Positivo',0)/len(out):.1%}")

            
            # prepare counts with clear column names to avoid Plotly confusion
            dept_counts = out["pred_department"].value_counts().reset_index(name="Conteggio")
            dept_counts.columns = ["Reparto", "Conteggio"]
            fig1 = px.bar(dept_counts,
                          x="Reparto", y="Conteggio",
                          title="Distribuzione Reparti")

            sent_counts_df = out["pred_sentiment"].value_counts().reset_index(name="Conteggio")
            sent_counts_df.columns = ["Sentiment", "Conteggio"]
            fig2 = px.bar(sent_counts_df,
                          x="Sentiment", y="Conteggio",
                          title="Distribuzione Sentiment", color="Sentiment")

            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

            if st.button("Esporta risultati"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"predictions_{timestamp}.csv"
                out.to_csv(fname, index=False)
                st.success(f"File salvato come {fname}")


if __name__ == "__main__":
    main()

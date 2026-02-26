import pandas as pd
import random
from datetime import datetime, timedelta

def generate_hotel_data(n=1000, include_neutral=False):


    reparti = ['Housekeeping', 'Reception', 'Food & Beverage']
    # sentiment binario
    if include_neutral:
        sentiments = ['Positivo', 'Negativo', 'Neutro']
    else:
        sentiments = ['Positivo', 'Negativo']
    
    # Liste di parole chiave ampie per ogni reparto per diversificare il dataset
    keywords = {
        'Housekeeping': [
            'pulizia', 'lenzuola', 'bagno', 'asciugamani', 'polvere', 'camera', 'ordine',
            'moquette', 'letto', 'minibar', 'cuscini', 'materasso', 'profumo', 'igiene',
            'sanitari', 'cambio biancheria', 'odore di chiuso', 'balcone', 'rifacimento letti',
            'aspirapolvere', 'armadio', 'tenda', 'scarsità di sapone', 'specchio',
            'scopa', 'spazzatura', 'tende', 'condizionatore', 'finestre', 'maniglie'
        ],
        'Reception': [
            'check-in', 'personale', 'accoglienza', 'chiavi', 'prenotazione', 'attesa',
            'receptionist', 'concierge', 'deposito bagagli', 'cortesia', 'efficienza',
            'informazioni', 'hall', 'lobby', 'disponibilità dello staff', 'gentilezza',
            'sorriso', 'orario di apertura', 'telefono', 'portiere', 'documenti',
            'gestione prenotazione', 'wifi', 'mappe', 'servizio clienti', 'lift',
            'navetta', 'indirizzo', 'modulo di check-in'
        ],
        'Food & Beverage': [
            'colazione', 'ristorante', 'cibo', 'caffè', 'cena', 'buffet', 'cameriere',
            'brioche', 'succo', 'qualità degli ingredienti', 'servizio ai tavoli', 'menu',
            'piatti', 'chef', 'tavolo', 'scelta dei vini', 'dolci', 'colazione a buffet',
            'carte dei vini', 'porzioni', 'sughi', 'antipasti', 'cucina', 'ristorazione',
            'bar', 'aperitivi', 'ristorante di qualità', 'conti errati'
        ]
    }

    data = []
    for i in range(n):
        rep = random.choice(reparti)
        sent = random.choice(sentiments)
        kw = random.choice(keywords[rep])
        
        
        review_id = f"REV-2025-{i+1:05d}"
        days_ago = random.randint(0, 90)
        review_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        
# generiamo titolo e corpo separati 
        if sent == 'Positivo':
            title_templates = [
                "Soggiorno fantastico", "Esperienza eccellente", "Personale impeccabile",
                "Consigliatissimo", "Tutto perfetto", "Altamente raccomandato"
            ]
            body_templates = [
                f"Soggiorno fantastico, la {kw} era eccellente e curata nei minimi dettagli.",
                f"Ho apprezzato molto la {kw}, servizio davvero professionale e impeccabile.",
                f"Tutto perfetto, un plauso particolare va alla {kw} deliziosa.",
                f"Siamo rimasti piacevolmente colpiti dalla {kw}, torneremo sicuramente.",
                f"La {kw} è stata superiore alle nostre aspettative, bravi!",
                f"Esperienza da 10 e lode per quanto riguarda la {kw}."
            ]
        elif sent == 'Negativo':
            title_templates = [
                "Esperienza pessima", "Non ci tornerò", "Servizio inaccettabile",
                "Deluso", "Problemi seri", "Pessimo rapporto qualità/prezzo"
            ]
            body_templates = [
                f"Pessima esperienza, la {kw} lasciava molto a desiderare e non era accettabile.",
                f"Sono rimasto deluso dalla {kw}, servizio scadente e poco attento al cliente.",
                f"Non tornerò mai più, ho riscontrato problemi seri con la {kw} del tutto assente.",
                f"Purtroppo la {kw} è stata il punto debole del nostro soggiorno, pessima.",
                f"Sconsigliato, la {kw} era indecente per un hotel di questo livello.",
                f"Gestione della {kw} totalmente da rivedere, molto frustrante."
            ]
        else: # Neutro
            title_templates = [
                "Soggiorno nella media", "Niente di speciale", "Adeguato",
                "Risultato prevedibile", "Giusto equilibrio", "Qualcosa da migliorare"
            ]
            body_templates = [
                f"Nella norma, la {kw} era ok, niente di eccezionale o memorabile.",
                f"Soggiorno senza infamia e senza lode, la {kw} era comunque accettabile.",
                f"La {kw} era media, adeguata al prezzo pagato per la stanza.",
                f"Niente di speciale la {kw}, rientra negli standard della categoria.",
                f"Soggiorno discreto, la {kw} potrebbe essere migliorata ma non è male.",
                f"La {kw} è stata ordinaria, conforme alla descrizione."
            ]
        
        title = random.choice(title_templates)
        body = random.choice(body_templates)
        data.append({
            'review_id': review_id,
            'date': review_date,
            'title': title,
            'body': body,
            'department': rep,
            'sentiment': sent
        })
    
    df = pd.DataFrame(data)
    # salviamo in un file csv all'interno della stessa cartella del progetto
    df.to_csv('hotel_reviews.csv', index=False, columns=['review_id','date','title','body','department','sentiment'])
    print(f"✅ Dataset generato correttamente con {n} recensioni in 'hotel_reviews.csv'")

if __name__ == "__main__":

    generate_hotel_data()

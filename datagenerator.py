import pandas as pd
import random
import os

def generate_hotel_data(n=600):
    reparti = ['Housekeeping', 'Reception', 'Food & Beverage']
    sentiments = ['Positivo', 'Negativo', 'Neutro']
    
    # Parole chiave per ogni reparto
    keywords = {
        'Housekeeping': ['pulizia', 'lenzuola', 'bagno', 'asciugamani', 'polvere', 'camera', 'ordine'],
        'Reception': ['check-in', 'personale', 'accoglienza', 'chiavi', 'prenotazione', 'attesa', 'receptionist'],
        'Food & Beverage': ['colazione', 'ristorante', 'cibo', 'caffè', 'cena', 'buffet', 'cameriere']
    }

    data = []
    for _ in range(n):
        rep = random.choice(reparti)
        sent = random.choice(sentiments)
        kw = random.choice(keywords[rep])
        
        # Generazione testo basata sul sentiment
        if sent == 'Positivo':
            templates = [
                f"Soggiorno fantastico, la {kw} era eccellente.",
                f"Ho apprezzato molto la {kw}, servizio impeccabile.",
                f"Tutto perfetto, specialmente la {kw}."
            ]
        elif sent == 'Negativo':
            templates = [
                f"Pessima esperienza, la {kw} lasciava molto a desiderare.",
                f"Sono rimasto deluso dalla {kw}, servizio scadente.",
                f"Non tornerò, problemi seri con la {kw}."
            ]
        else: # Neutro
            templates = [
                f"Nella norma, la {kw} era ok, niente di eccezionale.",
                f"Soggiorno senza infamia e senza lode, {kw} accettabile.",
                f"La {kw} era media, adeguata al prezzo."
            ]
            
        text = random.choice(templates)
        data.append({'review': text, 'department': rep, 'sentiment': sent})
    
    # Salvataggio
    df = pd.DataFrame(data)
    df.to_csv('hotel_reviews.csv', index=False)
    print(f"✅ Dataset generato con {n} recensioni in 'hotel_reviews.csv'")

if __name__ == "__main__":
    generate_hotel_data()
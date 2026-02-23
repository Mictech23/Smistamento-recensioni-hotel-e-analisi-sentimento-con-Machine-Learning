import random
import pandas as pd
import uuid

housekeeping_keywords = [
    "pulizia", "camera", "lenzuola", "asciugamani", "spazzare", "rifare la stanza"
]
reception_keywords = [
    "check-in", "check out", "reception", "accoglienza", "prenotazione", "receptionist"
]
fnb_keywords = [
    "colazione", "ristorante", "bar", "cena", "pranzo", "menu", "servizio in camera"
]

positive_phrases = [
    "è stato ottimo", "mi è piaciuto", "molto buono", "eccellente", "consigliato", "perfetto"
]
negative_phrases = [
    "è stato pessimo", "non mi è piaciuto", "molto cattivo", "scarso", "da evitare", "problemi"
]


def make_sentence(dept):
    
    if dept == "Housekeeping":
        kw = random.choice(housekeeping_keywords)
    elif dept == "Reception":
        kw = random.choice(reception_keywords)
    else:
        kw = random.choice(fnb_keywords)

    other_kw = []
    if dept != "Housekeeping":
        other_kw.extend(housekeeping_keywords)
    if dept != "Reception":
        other_kw.extend(reception_keywords)
    if dept != "Food & Beverage":
        other_kw.extend(fnb_keywords)

    text = kw
    if random.random() < 0.2 and other_kw:
        text += " e " + random.choice(other_kw)
    return text


def generate_reviews(n=1000, seed=42):
    random.seed(seed)
    reviews = []
    depts = ["Housekeeping", "Reception", "Food & Beverage"]

    dept_cycle = (depts * ((n // len(depts)) + 1))[:n]
    sentiments = (["Positivo"] * (n // 2)) + (["Negativo"] * (n - n // 2))
    random.shuffle(dept_cycle)
    random.shuffle(sentiments)

    for i in range(n):
        dept = dept_cycle[i]
        sentiment = sentiments[i]
        title = make_sentence(dept).capitalize()
        body = make_sentence(dept)
        if sentiment == "Positivo":
            body += " " + random.choice(positive_phrases)
        else:
            body += " " + random.choice(negative_phrases)

        reviews.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "body": body,
            "department": dept,
            "sentiment": sentiment,
        })
    df = pd.DataFrame(reviews)
    return df


if __name__ == "__main__":
    df = generate_reviews(1000)
    df.to_csv("hotel_reviews.csv", index=False)
    print("Dataset hotel_reviews.csv generato con successo (1000 record).")

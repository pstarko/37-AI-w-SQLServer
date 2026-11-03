import pyodbc
from sentence_transformers import SentenceTransformer, util

# Konfiguracja bazy SQL Server (dopasuj parametry do swojej instancji)
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=STARKOGABINET\SQLSERVER2025;"   # lub "NAZWA_SERWERA\INSTANCJA"
    "DATABASE=AI;"
    "UID=python;"      
    "PWD=oracle123;"              # hasło do SQL Server
)

# Model embeddingów
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Ładowanie modelu {model_name}... (chwila cierpliwości)")
model = SentenceTransformer(model_name)
print("Model załadowany.")

# Prototypy
prototype_sick = """
Pacjent ma wysoką gorączkę, , uporczywy kaszel, silne bóle mięśni i dreszcze.
Wyniki badań krwi są nieprawidłowe, podwyższone markery stanu zapalnego.
Samopoczucie bardzo złe, trudności z oddychaniem.
"""

prototype_healthy = """
Pacjent czuje się dobrze, brak gorączki, brak kaszlu ani bólu.
Badania krwi w normie, brak oznak stanu zapalnego.
Aktywny fizycznie, wysypia się, nie zgłasza istotnych dolegliwości.
Ma dobry apetyt i śpi dobrze. Nie pije alkoholu.
"""

# Wstępne embeddingi prototypów (raz na start)
emb_sick = model.encode(prototype_sick, convert_to_tensor=True)
emb_healthy = model.encode(prototype_healthy, convert_to_tensor=True)


def classify_patient(desc: str, threshold: float = 0.05):
    emb_patient = model.encode(desc, convert_to_tensor=True)
    sim_sick = util.cos_sim(emb_patient, emb_sick).item()
    sim_healthy = util.cos_sim(emb_patient, emb_healthy).item()

    if sim_sick - sim_healthy > threshold:
        return "chory", sim_sick, sim_healthy
    if sim_healthy - sim_sick > threshold:
        return "zdrowy", sim_sick, sim_healthy
    return "niejednoznaczne", sim_sick, sim_healthy


def fetch_patients(conn):
    cur = conn.cursor()
    cur.execute("SELECT patient_id, description FROM patients_100")
    rows = cur.fetchall()
    cur.close()
    return rows


def update_label(conn, patient_id, label: str):
    value = 1 if label == "chory" else 0 
    cur = conn.cursor()
    cur.execute(
        "UPDATE patients_100 SET is_sick = ? WHERE patient_id = ?",
        (value, patient_id),
    )
    cur.close()


if __name__ == "__main__":
    # Połączenie do bazy SQL Server
    conn = pyodbc.connect(CONN_STR)
    print("Połączono z bazą SQL Server")

    print("\nRozpoczynam klasyfikację pacjentów z tabeli 'patients_100'...\n")
    patients = fetch_patients(conn)
    for patient_id, desc in patients:
        label, sim_sick, sim_healthy = classify_patient(desc)
        update_label(conn, patient_id, label)
        print(f"=== {patient_id} ===")
        print(f"Opis: {desc}")
        print(f"Podobieństwo do 'chorego'   : {sim_sick:.3f}")
        print(f"Podobieństwo do 'zdrowego' : {sim_healthy:.3f}")
        print(f"Decyzja: {label} (zapisano {0 if label == 'chory' else 1} w from_model)\n")

    conn.commit()
    conn.close()

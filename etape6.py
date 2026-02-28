import pickle
import numpy as np

# Charger le modèle et le scaler
with open("modele_pannes.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predire_panne(type_machine, temperature_air, temperature_process, vitesse, couple, usure):
    # Encoder le type de machine
    type_encode = {'L': 0, 'M': 1, 'H': 2}[type_machine]
    
    # Créer les données
    donnees = np.array([[type_encode, temperature_air, temperature_process, vitesse, couple, usure]])
    
    # Normaliser
    donnees_scaled = scaler.transform(donnees)
    
    # Prédire
    prediction = model.predict(donnees_scaled)[0]
    probabilite = model.predict_proba(donnees_scaled)[0][1] * 100
    
    print(f"\n📊 Probabilité de panne : {probabilite:.1f}%")
    
    if prediction == 1:
        print("🔴 ALERTE — Panne probable ! Intervention recommandée.")
    else:
        print("🟢 Machine en bon état.")

# --- Teste avec ces exemples ---

print("=== Machine normale ===")
predire_panne(
    type_machine='M',
    temperature_air=300,
    temperature_process=310,
    vitesse=1500,
    couple=40,
    usure=100
)

print("\n=== Machine à risque ===")
predire_panne(
    type_machine='L',
    temperature_air=304,
    temperature_process=313,
    vitesse=1200,
    couple=70,
    usure=240
)
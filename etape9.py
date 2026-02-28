import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Features
X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Noms des zones de pannes
zones = {
    'TWF': '🔧 Usure outil',
    'HDF': '🌡️  Surchauffe',
    'PWF': '⚡ Surcharge électrique',
    'OSF': '⚙️  Contrainte mécanique',
    'RNF': '❓ Panne aléatoire'
}

# Entraîner un modèle par zone
modeles_zones = {}
print("=== Entraînement des modèles par zone ===\n")

for col, nom in zones.items():
    y = df[col]
    
    if y.sum() < 10:
        print(f"{nom} → pas assez de données, ignoré")
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
    
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train_bal, y_train_bal)
    
    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= 0.3).astype(int)
    
    from sklearn.metrics import recall_score
    recall = recall_score(y_test, y_pred)
    print(f"{nom} → Recall: {recall:.0%} ({y.sum()} pannes dans le dataset)")
    
    modeles_zones[col] = model

# Sauvegarder tous les modèles
with open("modeles_zones.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Modèles zones sauvegardés !")

# Test de prédiction complète
print("\n=== Test de prédiction complète ===")

def predire_complet(type_machine, temp_air, temp_process, vitesse, couple, usure):
    type_encode = {'L': 0, 'M': 1, 'H': 2}.get(type_machine, type_machine)
    ecart_temp = temp_process - temp_air
    donnees = np.array([[type_encode, temp_air, temp_process, vitesse, couple, usure, ecart_temp]])
    donnees_scaled = scaler.transform(donnees)
    
    # Charger le modèle principal
    with open("modele_pannes.pkl", "rb") as f:
        model_principal = pickle.load(f)
    
    proba_panne = model_principal.predict_proba(donnees_scaled)[0][1] * 100
    
    print(f"\n📊 Probabilité de panne : {proba_panne:.1f}%")
    
    if proba_panne >= 22:
        print("🔴 ALERTE — Panne probable !")
        print("\n📍 Zones à risque :")
        trouve = False
        for col, nom in zones.items():
            if col in modeles_zones:
                proba_zone = modeles_zones[col].predict_proba(donnees_scaled)[0][1] * 100
                if proba_zone >= 30:
                    print(f"   → {nom} : {proba_zone:.1f}% de risque")
                    trouve = True
        if not trouve:
            print("   → Zone non identifiée clairement")
    else:
        print("🟢 Machine en bon état")

# Machine normale
print("-- Machine normale --")
predire_complet('M', 300, 310, 1500, 40, 100)

# Machine à risque thermique
print("\n-- Machine à risque thermique --")
predire_complet('L', 304, 313, 1200, 70, 240)
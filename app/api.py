from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Prédiction d'Attrition (Churn)",
    description="API servant le modèle de Machine Learning pour prédire le risque de départ client.",
    version="1.0.0"
)

# Chargement du modèle et du preprocessor au démarrage de l'API
# Sécurité : les fichiers .joblib sont des artefacts entraînés et versionnés dans ce dépôt.
# Ils ne proviennent pas d'une source externe non contrôlée — chargement considéré sûr.
preprocessor = None
model = None
try:
    preprocessor = joblib.load('notebooks/preprocessor.joblib')
    model = joblib.load('notebooks/modele_logreg.joblib')
    print("Modèles chargés avec succès.")
except Exception as e:
    print(f"AVERTISSEMENT : Erreur de chargement des modèles : {e}")

# Définition du format attendu en entrée (basé sur le formulaire Streamlit)
class ClientFeatures(BaseModel):
    age: int
    contract_type: str
    monthly_fee: float
    tenure_months: int
    support_tickets: int
    avg_resolution_time: float
    payment_failures: int
    csat_score: int
    payment_method: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de Churn. Utilisez /docs pour voir le Swagger."}

@app.get("/health")
def health_check():
    """Route pour vérifier que l'API est en ligne (Health Check)."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(client: ClientFeatures):
    """
    Reçoit les données du client, applique le preprocessing et retourne la probabilité de churn.
    """
    try:
        # 1. Convertir la requête JSON en dictionnaire
        client_data = client.model_dump()
        
        # 2. Ajout de valeurs par défaut pour les variables manquantes (celles non demandées dans le Streamlit)
        # En production, on injecterait des médianes ou des modes calculés lors de l'EDA
        default_values = {
            'monthly_logins': 15, 'weekly_active_days': 3, 'avg_session_time': 30,
            'features_used': 10, 'usage_growth_rate': 0.0, 'last_login_days_ago': 14,
            'total_revenue': client.monthly_fee * client.tenure_months, # Logique métier
            'escalations': 0, 'email_open_rate': 0.5, 'marketing_click_rate': 0.1,
            'nps_score': 0, 'referral_count': 0, 'gender': 'Female', 'country': 'France',
            'city': 'Paris', 'customer_segment': 'Standard', 'signup_channel': 'Organic',
            'discount_applied': 'No', 'price_increase_last_3m': 'No', 
            'complaint_type': 'None', 'survey_response': 'No',
            'engagement_score': 0.43 # Moyenne calculée lors du feature engineering
        }
        
        # Fusionner les données reçues avec les valeurs par défaut
        full_data = {**default_values, **client_data}
        
        # 3. Créer un DataFrame pandas avec 1 seule ligne
        df_input = pd.DataFrame([full_data])
        
        # 4. Appliquer le preprocessor (Standardisation + OneHotEncoding)
        X_processed = preprocessor.transform(df_input)
        
        # 5. Prédire la probabilité (classe 1 = Churn)
        proba = model.predict_proba(X_processed)[0][1]
        prediction = int(model.predict(X_processed)[0])
        
        return {
            "prediction": prediction, # 0 ou 1
            "probability": float(proba), # ex: 0.78
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
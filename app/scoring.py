"""
Module de scoring partagé.

Centralise le feature engineering (engagement_score) et le scoring batch
de la base clients à partir des artefacts sérialisés (preprocessor + modèle).

Utilisé par le dashboard pour la vue "Analyse Globale" (scoring des 10 000 clients).
Le scoring client-à-client en temps réel passe, lui, par l'API FastAPI.
"""
from pathlib import Path
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# Racine du projet (app/ -> projet)
ROOT = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = ROOT / "notebooks" / "preprocessor.joblib"
MODEL_PATH = ROOT / "notebooks" / "modele_logreg.joblib"
DATA_PATH = ROOT / "data" / "raw" / "customer_churn_business_dataset.csv"

# Variables servant au calcul du score d'engagement (identiques au notebook)
ENGAGEMENT_COLS = [
    "monthly_logins", "weekly_active_days", "avg_session_time",
    "features_used", "usage_growth_rate", "last_login_days_ago",
]


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne 'engagement_score' (normalisation MinMax + pondération métier).

    Reproduit exactement la formule appliquée lors de la modélisation
    (notebooks/modelisation.ipynb) afin de garantir la cohérence train/serving.
    """
    df = df.copy()
    df_norm = df[ENGAGEMENT_COLS].copy()
    for col in ENGAGEMENT_COLS:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        if mx - mn > 0:
            df_norm[col] = (df_norm[col] - mn) / (mx - mn)
    df["engagement_score"] = (
        0.25 * df_norm["monthly_logins"]
        + 0.20 * df_norm["weekly_active_days"]
        + 0.20 * df_norm["avg_session_time"]
        + 0.10 * df_norm["features_used"]
        + 0.10 * df_norm["usage_growth_rate"]
        + 0.10 * (1.0 - df_norm["last_login_days_ago"])
    )
    return df


def load_artifacts():
    """Charge le preprocessor et le modèle sérialisés."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def load_raw_data() -> pd.DataFrame:
    """Charge le dataset clients brut."""
    return pd.read_csv(DATA_PATH)


def score_base(df: pd.DataFrame = None) -> pd.DataFrame:
    """Score la base clients et renvoie le DataFrame enrichi.

    Colonnes ajoutées :
      - churn_proba       : probabilité de résiliation [0, 1]
      - churn_pred        : prédiction binaire (seuil 0.5)
      - revenue_at_risk   : monthly_fee * churn_proba (manque à gagner mensuel attendu)
    """
    if df is None:
        df = load_raw_data()
    preprocessor, model = load_artifacts()

    work = df.copy()
    if "customer_id" in work.columns:
        work = work.drop(columns=["customer_id"])
    work = add_engagement_score(work)

    features = work.drop(columns=["churn"]) if "churn" in work.columns else work
    X = preprocessor.transform(features)
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = (proba >= 0.5).astype(int)
    out["revenue_at_risk"] = out["monthly_fee"] * out["churn_proba"]
    return out


def feature_importance() -> pd.DataFrame:
    """Renvoie l'importance des variables du modèle final (coefficients LogReg).

    Pour une régression logistique, l'amplitude du coefficient (sur features
    standardisées) traduit l'influence de la variable ; le signe en donne le sens
    (positif => pousse vers le churn).
    """
    preprocessor, model = load_artifacts()

    num_cols = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = preprocessor.transformers_[1][2]
    cat_features = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = list(num_cols) + cat_features

    coefs = model.coef_[0]
    imp = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "importance": abs(coefs),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp

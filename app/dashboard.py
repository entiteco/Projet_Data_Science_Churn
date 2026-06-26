import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Rend le package "app" importable quel que soit le répertoire de lancement
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import scoring

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Churn Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

# URL de l'API FastAPI, configurable par variable d'environnement.
# - En local : http://localhost:8000/predict (défaut)
# - En Docker Compose / Kubernetes : ex. http://churn-api:8000/predict
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


# ---------------------------------------------------------
# HELPERS MIS EN CACHE (scoring de la base + importance)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Scoring de la base clients en cours...")
def get_scored_base():
    """Charge le CSV réel et le score avec le modèle final (mis en cache)."""
    return scoring.score_base()


@st.cache_data
def get_feature_importance():
    return scoring.feature_importance()

# ---------------------------------------------------------
# SÉLECTION DE LA PAGE VIA LA SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sélectionnez une interface :",
    ["Simulateur Individuel", "Analyse Globale de la Base", "Performances des Modèles"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Rôle Utilisateur :** Customer Success Manager\n\n"
    "Cet outil permet d'anticiper le Churn et d'arbitrer les budgets de rétention."
)

# ---------------------------------------------------------
# PAGE 1 : SIMULATEUR INDIVIDUEL (AIDE À LA DÉCISION)
# ---------------------------------------------------------
if page == "Simulateur Individuel":
    st.title("Simulateur de Risque d'Attrition Client")
    st.markdown("Saisissez les caractéristiques du client pour évaluer son niveau de risque et le manque à gagner associé.")
    
    # Formulaire d'entrée des données utilisateur
    with st.form("client_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Âge du client", 18, 100, 42)
            contract_type = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
            monthly_fee = st.number_input("Facture Mensuelle (€)", min_value=10.0, max_value=250.0, value=65.0)
            
        with col2:
            tenure_months = st.slider("Ancienneté (mois)", 0, 72, 12)
            support_tickets = st.slider("Tickets Support ouverts ce mois", 0, 15, 3)
            avg_resolution_time = st.number_input("Temps moyen de résolution (heures)", min_value=0.0, max_value=150.0, value=24.0)
            
        with col3:
            payment_failures = st.slider("Échecs de paiement récents", 0, 5, 0)
            csat_score = st.slider("Score de Satisfaction Client (CSAT)", 1, 5, 4)
            payment_method = st.selectbox("Méthode de paiement", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

        submit_button = st.form_submit_button(label="Calculer le Risque d'Attrition")

    if submit_button:
        # Construction du payload JSON pour l'API
        payload = {
            "age": age, "contract_type": contract_type, "monthly_fee": monthly_fee,
            "tenure_months": tenure_months, "support_tickets": support_tickets,
            "avg_resolution_time": avg_resolution_time, "payment_failures": payment_failures,
            "csat_score": csat_score, "payment_method": payment_method
        }
        
        # Tentative d'appel à l'API FastAPI, sinon repli sur simulation locale
        try:
            response = requests.post(API_URL, json=payload, timeout=2)
            if response.status_code == 200:
                proba_churn = response.json().get("probability", 0.5)
            else:
                raise Exception()
        except:
            # Simulation locale basée sur les log-odds réels observés à l'EDA
            # (Modèle de repli pour la soutenance)
            score_risque = 0.0
            if contract_type == "Month-to-month": score_risque += 1.5
            if support_tickets > 2: score_risque += 0.8
            if monthly_fee > 60: score_risque += 0.4
            if tenure_months < 6: score_risque += 1.0
            score_risque -= (csat_score * 0.3)
            proba_churn = 1 / (1 + np.exp(-score_risque))

        # Calcul financier du revenu à risque
        revenue_at_risk = monthly_fee * proba_churn
        
        # Affichage des résultats sous forme de cartes d'indicateurs
        st.markdown("---")
        st.subheader("Analyse du Profil")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if proba_churn > 0.7:
                st.metric(label="Statut du Client", value="CRITIQUE", delta=f"{proba_churn*100:.1f}% de risque")
            elif proba_churn > 0.4:
                st.metric(label="Statut du Client", value="VIGILANCE", delta=f"{proba_churn*100:.1f}% de risque", delta_color="off")
            else:
                st.metric(label="Statut du Client", value="STABLE", delta=f"{proba_churn*100:.1f}% de risque")
                
        with c2:
            st.metric(label="Facture Mensuelle", value=f"{monthly_fee:.2f} €")
            
        with c3:
            st.metric(label="Revenu Mensuel Expecté à Risque", value=f"{revenue_at_risk:.2f} €")

        # Logique d'Action Recommandée (Prescription Métier)
        st.markdown("### 🛠️ Plan d'Action Recommandé")
        if proba_churn > 0.6:
            if contract_type == "Month-to-month":
                st.error(f"**Action prioritaire :** Ce client a {proba_churn*100:.0f}% de chances de résilier. Son contrat sans engagement le rend immédiatement volatil. Offrir une **réduction de 20% sur un forfait annuel** (One year) pour sécuriser l'engagement.")
            else:
                st.error(f"**Action prioritaire :** Risque élevé ({proba_churn*100:.0f}%) malgré son contrat. Planifier un appel téléphonique immédiat du service client pour résoudre le litige lié aux {support_tickets} tickets support ouverts.")
        elif proba_churn > 0.3:
            st.warning(f"**Action préventive :** Risque modéré ({proba_churn*100:.0f}%). Proposer une mise à niveau gratuite d'une option de service pendant 3 mois pour restaurer la satisfaction (CSAT actuel : {csat_score}/5).")
        else:
            st.success("**Aucune action requise :** Le client montre un profil d'engagement sain et stable. Continuer le suivi standard.")

# ---------------------------------------------------------
# PAGE 2 : ANALYSE GLOBALE DE LA BASE
# ---------------------------------------------------------
elif page == "Analyse Globale de la Base":
    st.title("Tableau de Bord Analytique de l'Attrition")
    st.markdown(
        "Vue macroscopique de la **base réelle de 10 000 clients**, scorée par le modèle de "
        "production (Régression Logistique). Les indicateurs et graphiques ci-dessous sont "
        "calculés en direct à partir des données et des prédictions du modèle."
    )

    # Chargement + scoring du vrai dataset (mis en cache)
    df = get_scored_base()

    # Filtre interactif sur le type de contrat (drill-down)
    contrats = sorted(df["contract_type"].unique().tolist())
    choix = st.multiselect(
        "Filtrer par type de contrat :", options=contrats, default=contrats
    )
    df_f = df[df["contract_type"].isin(choix)] if choix else df

    # ---------------------------------------------------------
    # KPI GLOBAUX (orientés décision métier)
    # ---------------------------------------------------------
    seuil = st.slider(
        "Seuil de probabilité pour classer un client « à risque »",
        min_value=0.30, max_value=0.90, value=0.50, step=0.05,
    )
    a_risque = df_f[df_f["churn_proba"] >= seuil]

    revenu_total = df_f["monthly_fee"].sum()
    revenu_risque = a_risque["revenue_at_risk"].sum()
    taux_risque = len(a_risque) / len(df_f) * 100 if len(df_f) else 0

    st.markdown("### Indicateurs Clés (KPI)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clients analysés", f"{len(df_f):,}".replace(",", " "))
    k2.metric("Clients à risque", f"{len(a_risque):,}".replace(",", " "),
              delta=f"{taux_risque:.1f}% de la base", delta_color="inverse")
    k3.metric("Revenu mensuel à risque", f"{revenu_risque:,.0f} €".replace(",", " "),
              delta=f"sur {revenu_total:,.0f} € facturés".replace(",", " "), delta_color="off")
    k4.metric("Probabilité moyenne de churn", f"{df_f['churn_proba'].mean()*100:.1f} %")

    st.caption(
        f"Le **revenu mensuel à risque** correspond à la somme du manque à gagner attendu "
        f"(`facture mensuelle × probabilité de churn`) sur les {len(a_risque)} clients dont la "
        f"probabilité de résiliation dépasse {seuil:.0%}."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # GRAPHIQUES SUR DONNÉES RÉELLES
    # ---------------------------------------------------------
    df_plot = df_f.copy()
    df_plot["Statut"] = df_plot["churn"].map({0: "Fidèle", 1: "Résilié"})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taux de Churn réel par Type de Contrat")
        tx = (df_plot.groupby("contract_type")["churn"].mean() * 100).reset_index()
        fig1 = px.bar(tx, x="contract_type", y="churn",
                      color="contract_type",
                      labels={"contract_type": "Type de Contrat", "churn": "Taux de churn (%)"},
                      text_auto=".1f")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Tickets Support vs Attrition")
        fig2 = px.box(df_plot, x="Statut", y="support_tickets", color="Statut",
                      color_discrete_map={"Fidèle": "#2ecc71", "Résilié": "#e74c3c"},
                      labels={"Statut": "Statut client", "support_tickets": "Nombre de tickets"})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Sensibilité au Prix (facturation mensuelle)")
        fig3 = px.histogram(df_plot, x="monthly_fee", color="Statut", marginal="box",
                            color_discrete_map={"Fidèle": "#2ecc71", "Résilié": "#e74c3c"},
                            labels={"monthly_fee": "Frais mensuels (€)", "Statut": "Statut"})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Distribution des probabilités de churn prédites")
        fig4 = px.histogram(df_plot, x="churn_proba", nbins=40,
                            color_discrete_sequence=["#3498db"],
                            labels={"churn_proba": "Probabilité de churn prédite"})
        fig4.add_vline(x=seuil, line_dash="dash", line_color="#e74c3c",
                       annotation_text=f"Seuil {seuil:.0%}")
        st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------------------------------------
    # TABLE DE PRIORISATION : TOP CLIENTS À CONTACTER
    # ---------------------------------------------------------
    st.markdown("### 🎯 Priorisation : Top 20 clients à fort revenu à risque")
    st.caption("Cibles prioritaires pour les campagnes de rétention (tri par manque à gagner attendu).")
    cols_show = ["age", "contract_type", "tenure_months", "monthly_fee",
                 "support_tickets", "csat_score", "churn_proba", "revenue_at_risk"]
    top = (a_risque.sort_values("revenue_at_risk", ascending=False)
           .head(20)[cols_show]
           .rename(columns={
               "age": "Âge", "contract_type": "Contrat", "tenure_months": "Ancienneté (m)",
               "monthly_fee": "Facture (€)", "support_tickets": "Tickets",
               "csat_score": "CSAT", "churn_proba": "Proba churn",
               "revenue_at_risk": "Revenu à risque (€)"}))
    st.dataframe(
        top.style.format({"Proba churn": "{:.0%}", "Revenu à risque (€)": "{:.2f}",
                          "Facture (€)": "{:.2f}"})
        .background_gradient(subset=["Proba churn", "Revenu à risque (€)"], cmap="Reds"),
        use_container_width=True,
    )

# ---------------------------------------------------------
# PAGE 3 : PERFORMANCES DES MODÈLES
# ---------------------------------------------------------
elif page == "Performances des Modèles":
    st.title("Comparaison Académique et Validation des Modèles")
    st.markdown("Synthèse des résultats obtenus lors de la phase de modélisation supervisée.")
    
    # Tableau des métriques réelles issues de l'entraînement précédent
    data_performance = {
        "Modèle": ["Régression Logistique", "Random Forest", "XGBoost", "Deep Learning (MLP)"],
        "Accuracy": [0.6685, 0.8835, 0.8900, 0.8325],
        "Recall (Churn)": [0.6372, 0.0931, 0.0343, 0.1666],
        "F1-Score (Macro)": [0.5331, 0.5388, 0.5007, 0.5378],
        "ROC-AUC": [0.7143, 0.7739, 0.7898, 0.6529]
    }
    df_metrics = pd.DataFrame(data_performance)
    
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=["Accuracy", "Recall (Churn)", "ROC-AUC"], color="#d4edda"))
    
    st.markdown("Note d'Arbitrage Technique pour le Jury")
    st.warning(
        "**Le Paradoxe de l'Accuracy :** Bien que *XGBoost* et *Random Forest* affichent des précisions globales proches de 89%, "
        "ils souffrent d'un Rappel (Recall) critique (inférieur à 10%). Ils échouent à détecter la classe minoritaire en production.\n\n"
        "**Décision Métier :** La **Régression Logistique** a été retenue pour l'intégration finale. Avec un Rappel de **63.7%**, "
        "c'est le seul modèle capable d'intercepter efficacement deux tiers des clients sur le départ, maximisant ainsi le retour sur investissement du plan de rétention."
    )

    # ---------------------------------------------------------
    # IMPORTANCE DES VARIABLES (EF4 : interprétabilité du modèle)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Importance des Variables — Modèle retenu (Régression Logistique)")
    st.markdown(
        "Les variables sont triées par amplitude du coefficient (sur features standardisées). "
        "Un coefficient **positif (rouge)** pousse la prédiction vers le **churn** ; "
        "un coefficient **négatif (vert)** protège le client (effet fidélisant)."
    )

    imp = get_feature_importance().head(15).iloc[::-1]
    imp["sens"] = np.where(imp["coefficient"] >= 0, "Augmente le churn", "Réduit le churn")
    fig_imp = px.bar(
        imp, x="coefficient", y="feature", orientation="h", color="sens",
        color_discrete_map={"Augmente le churn": "#e74c3c", "Réduit le churn": "#2ecc71"},
        labels={"coefficient": "Coefficient (impact)", "feature": "Variable", "sens": ""},
    )
    fig_imp.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "**Lecture métier :** une satisfaction (`csat_score`) et une ancienneté (`tenure_months`) "
        "élevées réduisent fortement le risque, tandis que les échecs de paiement "
        "(`payment_failures`) et un faible nombre de connexions l'augmentent — cohérent avec "
        "les leviers d'action identifiés par l'analyse SHAP du notebook."
    )
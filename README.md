# Système Intelligent Multi-Modèles — Rétention Client & Risque de Revenus

Projet Data Science certifiant **RNCP40875 Expert en Ingénierie de Données**
Bloc 2 – Pilotage et implémentation de solutions IA
**EFREI — Mastère DE3 M1 Data Engineering & IA 2025-2026**

Étudiants : **Kévin HEUGAS** & **Valentin MASSONNIERE**

---

## Présentation du projet

Dans le secteur des services, acquérir un nouveau client coûte 5 à 25 fois plus cher que d'en fidéliser un existant. Ce projet transforme des données d'activité clients brutes en un **système d'aide à la décision proactif** permettant aux équipes Customer Success et Finance d'anticiper le désengagement et d'évaluer dynamiquement le **"Revenu à Risque"**.

### Ce que fait la solution

| Composant                         | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| **API FastAPI**             | Micro-service d'inférence exposant le modèle ML (port 8000) |
| **Dashboard Streamlit**     | 3 pages : simulateur client (via API), analyse globale de la base **réelle** (KPI + revenu à risque), comparaison des modèles & importance des variables |
| **Notebooks Jupyter**       | EDA, preprocessing, entraînement, **validation croisée** et comparaison des modèles |
| **Docker / Docker-Compose** | Environnement conteneurisé, reproductible multiplateforme    |

### Modèles entraînés et comparés

| Modèle                | Accuracy | Recall (Churn)  | ROC-AUC         | Retenu        |
| ---------------------- | -------- | --------------- | --------------- | ------------- |
| Régression Logistique | 66.9%    | **63.7%** | 0.714           | **Oui** |
| Random Forest          | 88.4%    | 9.3%            | 0.774           | Non           |
| XGBoost                | 89.0%    | 3.4%            | **0.790** | Non           |
| Deep Learning (MLP)    | 83.3%    | 16.7%           | 0.653           | Non           |

> **Décision métier :** La Régression Logistique a été retenue malgré une accuracy inférieure. Avec un Recall de 63.7%, c'est le seul modèle capable de détecter deux tiers des clients sur le départ — ce qui maximise le ROI des campagnes de rétention.

---

## Architecture du projet

```
Projet_Data_Science_Churn/
├── app/
│   ├── api.py              # API FastAPI (endpoints /predict, /health)
│   └── dashboard.py        # Dashboard Streamlit (3 pages)
├── data/
│   └── raw/
│       └── customer_churn_business_dataset.csv   # 10 000 clients, 32 variables
├── notebooks/
│   ├── eda.ipynb           # Analyse Exploratoire des Données
│   ├── modelisation.ipynb  # Entraînement et comparaison des modèles
│   ├── modele_logreg.joblib    # Modèle sérialisé (Régression Logistique)
│   └── preprocessor.joblib    # Pipeline de preprocessing sérialisé
├── Dockerfile              # Image Python 3.10-slim
├── docker-compose.yml      # Services : jupyter (8888) + api (8000)
├── Makefile                # Raccourcis de commandes Docker
└── requirements.txt        # Dépendances Python
```

### Pipeline IA (bout en bout)

```
CSV brut
  └─► EDA (eda.ipynb)
        └─► Preprocessing (StandardScaler + OneHotEncoder + SMOTE train-only)
              └─► Entraînement multi-modèles + validation croisée 5-folds (modelisation.ipynb)
                    └─► Interprétabilité (SHAP + importance des variables)
                          └─► Sérialisation joblib (modele_logreg + preprocessor)
                                └─► API FastAPI (/predict)
                                      └─► Dashboard Streamlit (simulateur + KPI base réelle)
```

---

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) >= 24.x
- [Docker Compose](https://docs.docker.com/compose/) >= 2.x (inclus dans Docker Desktop)
- `make` (optionnel, disponible nativement sur macOS/Linux)

Pour une installation **locale sans Docker** :

- Python >= 3.10
- pip

---

## Installation et lancement

### Avec Docker (recommandé)

```bash
# 1. Cloner le dépôt
git clone <https://github.com/entiteco/Projet_Data_Science_Churn>
cd Projet_Data_Science_Churn

# 2. Construire les images
make build
# ou : docker-compose build

# 3. Lancer tous les services en arrière-plan
make up
# ou : docker-compose up -d
```

Les services sont accessibles à :

- **API FastAPI** → http://localhost:8000
- **Swagger UI** → http://localhost:8000/docs
- **Jupyter Notebook** → http://localhost:8888

### Commandes Makefile disponibles

| Commande         | Action                                   |
| ---------------- | ---------------------------------------- |
| `make build`   | Construit les images Docker              |
| `make up`      | Lance tous les services en arrière-plan |
| `make down`    | Arrête et supprime les conteneurs       |
| `make jupyter` | Lance uniquement Jupyter (interactif)    |
| `make api`     | Lance uniquement l'API (interactif)      |
| `make bash`    | Ouvre un terminal dans le conteneur      |

### Lancer le Dashboard Streamlit

Le dashboard n'est pas inclus dans docker-compose. Pour le lancer :

```bash
# En local (Python installé)
pip install -r requirements.txt
streamlit run app/dashboard.py

# Dans le conteneur Docker
docker-compose run --rm api streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Dashboard accessible sur → http://localhost:8501

### Sans Docker (environnement local)

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate       # macOS/Linux
# .\venv\Scripts\activate      # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# Lancer les notebooks (dans un second terminal)
jupyter notebook notebooks/
```

---

## API FastAPI — Référence des endpoints

### `GET /`

Vérification que l'API répond.

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

Retourne la probabilité de churn pour un client.

**Corps de la requête (JSON) :**

```json
{
  "age": 42,
  "contract_type": "Month-to-month",
  "monthly_fee": 65.0,
  "tenure_months": 12,
  "support_tickets": 3,
  "avg_resolution_time": 24.0,
  "payment_failures": 0,
  "csat_score": 4,
  "payment_method": "Electronic check"
}
```

**Réponse :**

```json
{
  "prediction": 1,
  "probability": 0.78,
  "status": "success"
}
```

> La documentation interactive complète est disponible sur http://localhost:8000/docs (Swagger UI).

---

## Dataset

| Attribut       | Valeur                                                  |
| -------------- | ------------------------------------------------------- |
| Fichier        | `data/raw/customer_churn_business_dataset.csv`        |
| Lignes         | 10 000 (clients)                                        |
| Colonnes       | 32 variables                                            |
| Variable cible | `churn` (binaire : 1 = résilié, 0 = actif)          |
| Déséquilibre | ~20% de churn (classe minoritaire) → traité par SMOTE |

**Variables principales utilisées par le modèle :**

| Variable             | Type        | Description                   |
| -------------------- | ----------- | ----------------------------- |
| `age`              | Numérique  | Âge du client                |
| `tenure_months`    | Numérique  | Ancienneté en mois           |
| `contract_type`    | Catégoriel | Monthly / One year / Two year |
| `monthly_fee`      | Numérique  | Facture mensuelle (€)        |
| `support_tickets`  | Numérique  | Tickets ouverts ce mois       |
| `csat_score`       | Numérique  | Score de satisfaction (1–5)  |
| `payment_failures` | Numérique  | Échecs de paiement récents  |
| `payment_method`   | Catégoriel | Mode de règlement            |

---

## Stack technique

| Couche              | Technologie              | Version          |
| ------------------- | ------------------------ | ---------------- |
| Langage             | Python                   | 3.10             |
| API                 | FastAPI + Uvicorn        | 0.103.1 / 0.23.2 |
| Dashboard           | Streamlit + Plotly       | ≥1.28 / ≥5.17  |
| ML                  | scikit-learn             | 1.4.2            |
| Boosting            | XGBoost                  | ≥1.7            |
| Rééchantillonnage | imbalanced-learn (SMOTE) | 0.11.0           |
| Interprétabilité  | SHAP                     | ≥0.43           |
| Sérialisation      | joblib                   | 1.3.2            |
| Conteneurisation    | Docker + Docker-Compose  | —               |

---

## Dépannage

**Les modèles `.joblib` ne se chargent pas**
Vérifier que les fichiers `notebooks/modele_logreg.joblib` et `notebooks/preprocessor.joblib` existent. Si absent, relancer le notebook `modelisation.ipynb` depuis Jupyter.

**Port déjà utilisé**

```bash
# Vérifier quel processus utilise le port 8000
lsof -i :8000
# Arrêter les conteneurs existants
make down
```

**Rebuild après modification du code**

```bash
make down && make build && make up
```

---

## Contexte académique

Ce projet a été réalisé dans le cadre de la certification **RNCP40875 Expert en Ingénierie de Données**, Bloc 2 — Pilotage et implémentation de solutions IA, à l'**EFREI Paris Panthéon-Assas Université**.

La méthodologie adoptée est hybride **Agile/CRISP-DM**, avec suivi des tâches via GitHub Projects et gestion des dépendances cross-plateforme (macOS/Windows) via `.gitattributes` et Docker.

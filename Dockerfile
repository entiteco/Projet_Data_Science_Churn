# 1. Utiliser une image Python officielle légère
# Python 3.12 : requis par scikit-learn 1.8 (qui a abandonné Python 3.10),
# pour que la version installée corresponde exactement à celle des artefacts .joblib.
FROM python:3.12-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier le fichier des dépendances
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le code de l'API et les modèles (.joblib)
COPY . .

# 6. Exposer le port 8000 pour FastAPI
EXPOSE 8000

# 7. Commande de lancement de l'API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
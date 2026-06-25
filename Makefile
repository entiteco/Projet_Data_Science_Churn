.PHONY: build up down jupyter api bash

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

# Lance juste Jupyter pour bosser sur les notebooks
jupyter:
	docker-compose up jupyter

# Lance l'API pour la tester
api:
	docker-compose up api

# Ouvre un terminal à l'intérieur du conteneur (très pratique pour débugger)
bash:
	docker-compose run --rm jupyter /bin/bash
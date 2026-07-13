.PHONY: up down build logs test migrate inspect build-engines health frontend-build

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f api

migrate:
	docker compose exec api alembic upgrade head

test:
	docker compose exec api pytest -q

inspect:
	docker compose exec api python scripts/inspect_models.py

build-engines:
	docker compose exec api python scripts/build_engines.py --if-needed

health:
	curl -f http://localhost:8000/health/live
	curl -f http://localhost:8000/health/ready

frontend-build:
	docker compose run --rm ui-build npm run build

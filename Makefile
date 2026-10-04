.PHONY: install train test api docker

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

test:
	pytest tests/ -v

api:
	uvicorn src.api.main:app --reload

docker:
	docker-compose -f docker/docker-compose.yml up --build
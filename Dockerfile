# Self-contained build for Stage 3 reproduction. Lets judges docker-pull and docker-run without manual environment setup.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY role_model.yaml .
COPY validate_submission.py .

ENTRYPOINT ["python"]

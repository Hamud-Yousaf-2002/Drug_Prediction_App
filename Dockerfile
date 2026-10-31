# Python 3.13: scikit-learn 1.6.1 (needed to load the saved model) has no wheels for newer versions
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py app.py drug_prediction_model.pkl ./

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501

# Default: run the API. docker-compose overrides this for the Streamlit UI.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN pip install --no-deps -e .

WORKDIR /app/src/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
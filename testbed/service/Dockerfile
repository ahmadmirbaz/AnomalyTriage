FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
# Threaded workers, single process: the fault threads and the metrics they
# move have to live in the same process the scraper reads.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "16", "app:app"]

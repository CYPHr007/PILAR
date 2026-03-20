FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD gunicorn etape7:app --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 120
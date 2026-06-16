FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y nodejs npm

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN cd frontend && npm install && npm run build

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
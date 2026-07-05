# Dockerfile
# Imagen base ligera
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define la carpeta interna del contenedor
WORKDIR /app

# Copia las dependencias
COPY requirements.txt .

# Instalar dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código y el modelo entrenado
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .

# Exponer puerto porque se usa una API
EXPOSE 8000

# Define el comando de ejecución
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Contenido para el archivo: app_streamlit/Dockerfile

# --- PASO 1: IMAGEN BASE ---
# Usamos una imagen oficial de Python 3.12, versión "slim" que es más ligera.
FROM python:3.12-slim

# --- PASO 2: DIRECTORIO DE TRABAJO ---
# Creamos y nos movemos a una carpeta llamada /app dentro del contenedor.
WORKDIR /app

# --- PASO 3: COPIAR ARCHIVOS ---
# Copiamos primero el archivo de dependencias para optimizar el caché de Docker.
COPY requirements.txt .

# --- PASO 4: INSTALAR DEPENDENCIAS DE SISTEMA ---
# OpenCV necesita algunas librerías del sistema operativo (Linux) para funcionar.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# --- PASO 5: INSTALAR LIBRERÍAS DE PYTHON ---
# Usamos pip para instalar todo lo que listamos en requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# --- PASO 6: COPIAR EL RESTO DE LA APLICACIÓN ---
# Copiamos todo el contenido de la carpeta app_streamlit al contenedor.
COPY . .

# --- PASO 7: EXPONER EL PUERTO ---
# Le decimos a Docker que el contenedor escuchará en el puerto 8501.
EXPOSE 8501

# --- PASO 8: COMANDO DE EJECUCIÓN ---
# Este es el comando para ejecutar la aplicación cuando el contenedor inicie.
# Usamos el nombre de tu script principal (ej. Presentacion.py o app.py).
CMD ["streamlit", "run", "Presentacion.py", "--server.port=8501", "--server.address=0.0.0.0"]
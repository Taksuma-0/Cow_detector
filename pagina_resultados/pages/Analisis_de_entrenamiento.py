import streamlit as st
from PIL import Image

st.set_page_config(page_title="Análisis de Entrenamiento", page_icon="📈", layout="wide")

st.title("📈 Análisis de Rendimiento de los Entrenamientos")
st.write("Aquí puedes visualizar y comparar las métricas de rendimiento de cada modelo durante su fase de entrenamiento.")


model_type = st.selectbox(
    "Selecciona el modelo para ver sus resultados de entrenamiento:",
    ("Detección de Objetos (YOLOv8)")
)

st.markdown("---")

if model_type == "Detección de Objetos (YOLOv8)":
    st.header("Resultados del Modelo de Detección")
    
    
    results_path = "training_results/detection/results.png"
    confusion_path = "training_results/detection/confusion_matrix.png"

    try:
        st.subheader("Curvas de Entrenamiento (Pérdida y mAP)")
        st.image(str(results_path))

        st.subheader("Matriz de Confusión")
        st.image(str(confusion_path))
        
    except FileNotFoundError:
        st.error(f"No se encontraron los archivos de resultados para el modelo de detección.")
        st.info(f"El script está buscando en: '{results_path.resolve()}'")
      
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Análisis de Entrenamiento", page_icon="📈", layout="wide")

st.title("📈 Análisis de Rendimiento de los Entrenamientos")
st.write("Aquí puedes visualizar y comparar las métricas de rendimiento de cada modelo durante su fase de entrenamiento.")


model_type = st.selectbox(
    "Selecciona el modelo para ver sus resultados de entrenamiento:",
    ("Detección de Objetos (YOLOv8)", "Segmentación de Instancias (YOLOv8)","Metricas ResNet50")
)

st.markdown("---")

if model_type == "Detección de Objetos (YOLOv8)":
    st.header("Resultados del Modelo de Detección")
    try:
        st.subheader("Curvas de Entrenamiento (Pérdida y mAP)")
        st.image("C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/training_results/detection/results.png")

        st.subheader("Matriz de Confusión")
        st.image("C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/training_results/detection/confusion_matrix.png")
        
    except FileNotFoundError:
        st.error("No se encontraron los archivos de resultados para el modelo de detección. Verifica que las rutas sean correctas y los archivos existan.")

elif model_type == "Segmentación de Instancias (YOLOv8)":
    st.header("Resultados del Modelo de Segmentación")
    try:
        st.subheader("Curvas de Entrenamiento (Pérdida y mAP)")
        st.image("C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/training_results/segmentation/results.png")
        
        st.subheader("Matriz de Confusión")
        st.image("C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/training_results/segmentation/confusion_matrix.png")
        
    except FileNotFoundError:
        st.error("No se encontraron los archivos de resultados para el modelo de segmentación. Verifica que las rutas sean correctas y los archivos existan.")


elif model_type == "Metricas ResNet50":
    st.header("Resultados")
    try:
        st.subheader("Matriz de Confusión")
        st.image("C:/Users/welin/OneDrive/Escritorio/cow_detector/Resnet_model/confusion_matrix.png")
        
    except FileNotFoundError:
        st.error("No se encontraron los archivos de resultados para el modelo de segmentación. Verifica que las rutas sean correctas y los archivos existan.")        
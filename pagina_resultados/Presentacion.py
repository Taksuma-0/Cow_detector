import streamlit as st


st.set_page_config(page_title="Presentación del Proyecto", page_icon="🚀", layout="wide")


st.title("Presentación: Sistema Inteligente para el Monitoreo de Ganado")
st.markdown("---")


st.header("1. Visión de Negocio: Ganadería de Precisión")
st.markdown(
    """
    La tecnología desarrollada aquí es la base para una potente solución comercial de **Ganadería de Precisión**.
    El objetivo es ofrecer a los ganaderos una herramienta que automatice la supervisión de sus rebaños,
    traduciéndose en ahorro de tiempo, reducción de costos operativos y una mejora sustancial en la salud y
    productividad del ganado.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Propuestas de Valor")
    st.success(
        """
        - **Conteo Automático de Reses:** Eliminar el conteo manual, propenso a errores, mediante un sistema automatizado que funciona con drones o cámaras fijas.
        - **Monitoreo de Salud y Comportamiento:** Gracias al tracking, podemos detectar anomalías. ¿Una vaca se ha aislado del rebaño? ¿No se ha movido en horas? Son indicadores tempranos de enfermedad.
        - **Optimización del Pastoreo:** Analizar qué zonas de un potrero son más utilizadas para optimizar la rotación y evitar el sobrepastoreo.
        - **Seguridad:** Alertas automáticas si el número de cabezas disminuye inesperadamente o si se detecta movimiento en zonas no autorizadas.
        """
    )


st.markdown("---")

st.header("2. Stack Tecnológico y Metodología")


st.subheader("📚 Datasets Empleados")
st.markdown(
    """
    Para entrenar estos modelos, se combinaron y curaron varios datasets públicos:
    - **Dataset de Vacas (Detección):** *Cows Detection Dataset* de Kaggle, para la clase positiva.
    - **Dataset de Vacas (Segmentación):** *Cow Instance Segment* de Roboflow, para la tarea de segmentación.
    - **Dataset de Negativos y Fondos:** Se utilizaron imágenes de los datasets *Intel Image Classification* (clase 'pasture') y *Animals-10* para enseñar al modelo a diferenciar el entorno y otros animales.
    """
)

st.subheader("🛠️ Infraestructura y Tecnologías")
st.markdown(
    """
    - **Lenguaje:** Python 3.12
    - **Deep Learning:** PyTorch
    - **Detección/Segmentación:** Ultralytics (YOLOv8)
    - **Visión por Computadora:** OpenCV
    - **Aplicación Web Interactiva:** Streamlit
    - **Gestión de Entorno:** Conda
    - **Hardware de Entrenamiento:** GPU NVIDIA GeForce GTX 1070 (8GB)
    """
)

st.subheader("Diagrama de infraestructura")
st.image("Diagrama.png")
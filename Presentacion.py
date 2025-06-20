import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Presentación del Proyecto", page_icon="🚀", layout="wide")


st.title("🚀 Presentación: Sistema Inteligente para el Monitoreo de Ganado")
st.markdown("---")


st.header("1. Introducción al Proyecto")
st.write(
    """
    Esta es la presentación de nuestro proyecto de Visión por Computadora diseñado para la detección,
    seguimiento y análisis de ganado bovino en tiempo real. Partiendo de una simple clasificación de imágenes,
    el sistema ha evolucionado para incorporar y comparar múltiples modelos de Deep Learning de última generación,
    incluyendo clasificación, detección de objetos y segmentación de instancias.

    El objetivo es demostrar una solución práctica y robusta que pueda ser la base para aplicaciones comerciales
    en el sector de la **agrotecnología (AgriTech)**.
    """
)

st.markdown("---")

st.header("2. Visión de Negocio: Ganadería de Precisión")
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

with col2:
    st.subheader("💰 Modelo de Negocio")
    st.info(
        """
        - **Producto:** Una plataforma web (SaaS - Software as a Service) donde los clientes suben sus videos (de drones, cámaras de seguridad) y reciben dashboards con analíticas.
        - **Mercado Objetivo:** Productores ganaderos de mediana y gran escala, cooperativas agrícolas y empresas de seguros agropecuarios.
        - **Monetización:** Un modelo de suscripción mensual basado en el número de cabezas de ganado o hectáreas a monitorear.
        """
    )

st.markdown("---")

st.header("3. Stack Tecnológico y Metodología")

st.subheader("🧠 Modelos de IA Utilizados")
st.markdown(
    """
    Se entrenaron y compararon tres arquitecturas para distintas tareas:
    - **Clasificación (ResNet50):** Un modelo base para determinar la presencia o ausencia de vacas en un fotograma completo.
    - **Detección de Objetos (YOLOv8-Detect):** Un modelo rápido y eficiente que dibuja **cajas delimitadoras** alrededor de cada vaca.
    - **Segmentación de Instancias (YOLOv8-Segment):** El modelo más avanzado, que delinea la **silueta exacta** de cada vaca, permitiendo análisis más precisos.
    """
)

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
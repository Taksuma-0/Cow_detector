import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd
import time
import torch


st.set_page_config(page_title="Dashboard de Detección de Vacas", page_icon="🐄", layout="wide")

# --- CONSTANTES Y MODELO ---
MODEL_PATH = Path("models/best.pt") 
TRACKER_CONFIG = "bytetrack.yaml"  


# Esta es la "memoria" que persiste entre interacciones del usuario
if 'cow_data' not in st.session_state:
    # Usaremos un diccionario para guardar los datos: {track_id: {'Dato': 'valor'}}
    st.session_state.cow_data = {}


def load_yolo_model(path):
    """Carga el modelo YOLOv8 una sola vez con manejo de errores."""
    # Se elimina @st.cache_resource si se necesita reiniciar el tracker cargando el modelo de nuevo.
    # En este flujo, reiniciaremos los datos en la sesión, por lo que el caché del modelo es aceptable.
    try:
        model = YOLO(path)
        #model.to(DEVICE)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en '{path}'.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def draw_annotations(frame, results,peso,highlight_id=None):
    """Dibuja las detecciones y añade datos personalizados a la etiqueta."""
    annotated_frame = frame.copy()
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]
            conf = confs[i]

            is_highlighted = (track_id == highlight_id)
            color = (255, 0, 255) if is_highlighted else (0, 255, 0) # Magenta si se destaca
            thickness = 3 if is_highlighted else 2
            
            # --- Etiqueta Dinámica con Datos Personalizados 
            
            label = f"Vaca #{track_id} ({conf:.2f}) - Peso {peso}[kg] " # Etiqueta base
                
            if track_id in st.session_state.cow_data:
                custom_data = st.session_state.cow_data[track_id].get('Dato', '')
                if custom_data:
                    label += f" - {custom_data}" # Añadimos el dato a la etiqueta
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return annotated_frame

def draw_annotations1(frame, results,highlight_id=None):
    """Dibuja las detecciones y añade datos personalizados a la etiqueta."""
    annotated_frame = frame.copy()
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]
            conf = confs[i]

            is_highlighted = (track_id == highlight_id)
            color = (255, 0, 255) if is_highlighted else (0, 255, 0) # Magenta si se destaca
            thickness = 3 if is_highlighted else 2
            
            
            label = f"Vaca #{track_id} ({conf:.2f})" # Etiqueta base
                
            if track_id in st.session_state.cow_data:
                custom_data = st.session_state.cow_data[track_id].get('Dato', '')
                if custom_data:
                    label += f" - {custom_data}" # Añadimos el dato a la etiqueta
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return annotated_frame



st.title("🐄 Dashboard de Gestión y Seguimiento de Ganado")
st.markdown("Sube un video, presiona 'Iniciar Análisis' y utiliza las herramientas de la barra lateral.")

st.sidebar.header("Configuración de Inferencia")
confidence_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.50, 0.05)
iou_threshold = st.sidebar.slider("Umbral de IoU (NMS)", 0.0, 1.0, 0.45, 0.05)
st.sidebar.markdown("---")
st.sidebar.header("Anotación de Datos")
with st.sidebar.form(key="data_form", clear_on_submit=True):
    annotation_id = st.number_input("ID de Vaca a Anotar", min_value=0, step=1)
    highlight_id_input = annotation_id
    annotation_data = st.text_input("Dato a registrar (ej. 'Peso: 550kg')")
    submit_button = st.form_submit_button(label="Guardar Dato")

    if submit_button and annotation_data:
        st.session_state.cow_data[annotation_id] = {'Peso (Kg)': annotation_data}
        st.sidebar.success(f"Dato guardado para la Vaca #{annotation_id}!")

st.sidebar.markdown("---")
st.sidebar.header("Datos Registrados")
df_data = pd.DataFrame() 

if st.session_state.cow_data:
    df_data = pd.DataFrame.from_dict(st.session_state.cow_data, orient='index')
    df_data.index.name = "ID Vaca"
    st.sidebar.dataframe(df_data)
else:
    st.sidebar.info("Aún no se han registrado datos.")
uploaded_file = st.sidebar.file_uploader("Carga tu video aquí", type=['mp4', 'mov', 'avi', 'mkv'])
start_button = st.sidebar.button("Iniciar / Reiniciar Análisis", type="primary")





model = load_yolo_model(MODEL_PATH)

if uploaded_file and start_button:
    # Reiniciar el tracker y los datos al iniciar un nuevo análisis
    if hasattr(model, 'reset_tracker'):
        model.reset_tracker() 
    else: # Si no, recargamos el modelo para limpiar el estado del tracker
        model = load_yolo_model(MODEL_PATH)

    st.session_state.cow_data = {} # Limpiamos los datos de la sesión anterior

    # Layout de la app
    video_col, kpi_col = st.columns([3, 1])
    with video_col:
        st.subheader("Video Procesado en Vivo")
        video_placeholder = st.empty()
    with kpi_col:
        st.subheader("Métricas en Vivo")
        kpi1_placeholder = st.empty()
        kpi2_placeholder = st.empty()
        kpi3_placeholder = st.empty()
        
    
    st.subheader("📈 Historial de Conteo de Vacas")
    chart_placeholder = st.empty()
    progress_bar_placeholder = st.empty()

    # Procesamiento del video
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    cow_counts_history = []
    frame_number = 0
    start_time = time.time() 

    if total_frames > 0:
        progress_bar = progress_bar_placeholder.progress(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        try:
            results = model.track(frame, conf=confidence_threshold, iou=iou_threshold, persist=True, tracker=TRACKER_CONFIG, verbose=False)
            
            if not df_data.empty:
                annotated_frame = draw_annotations(frame, results,df_data.loc[annotation_id, 'Peso (Kg)'],highlight_id_input if highlight_id_input > 0 else None)
                df_data = pd.DataFrame() 
                
            else:
                annotated_frame = draw_annotations1(frame, results, highlight_id_input if highlight_id_input > 0 else None)
            
            cow_count = len(results[0])
            
            cow_counts_history.append(cow_count)
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Actualizar Métricas en Vivo
            processing_time = time.time() - start_time
            fps = (frame_number + 1) / processing_time if processing_time > 0 else 0
            current_time_sec = frame_number / video_fps
            
            kpi1_placeholder.metric("Conteo Actual", f"{cow_count} Vacas")
            kpi2_placeholder.metric("Velocidad (FPS)", f"{fps:.2f}")
            kpi3_placeholder.metric("Tiempo de Video", f"{current_time_sec:.1f}s")
            
            
            df_chart = pd.DataFrame(cow_counts_history, columns=['Vacas Detectadas'])
            chart_placeholder.line_chart(df_chart)
            
            if total_frames > 0:
                progress = (frame_number + 1) / total_frames
                progress_bar_placeholder.progress(min(progress, 1.0))

        except Exception as e:
            st.warning(f"⚠️ Se encontró un fotograma problemático y fue omitido. (Error: {e})")
            continue
        
        frame_number += 1

    cap.release()
    st.success("Análisis de video finalizado.")
    # --- Resumen Final del Análisis ---
    with st.expander("📊 Ver Resumen Final del Análisis", expanded=True):
            if cow_counts_history:
                max_cows = int(np.max(cow_counts_history))
                avg_cows = np.mean(cow_counts_history)
                max_cow_frame = np.argmax(cow_counts_history)
                time_of_max = max_cow_frame / video_fps
                frames_with_cows = sum(1 for count in cow_counts_history if count > 0)
                percentage_with_cows = (frames_with_cows / len(cow_counts_history)) * 100
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Máximo de Vacas Detectadas", f"{max_cows}")
                res_col2.metric("Promedio de Vacas por Frame", f"{avg_cows:.2f}")
                res_col3.metric("Momento de Máx. Detección", f"{time_of_max:.1f}s")
                st.metric("% de Frames con Detecciones", f"{percentage_with_cows:.1f}%")

else:
    if not uploaded_file:
        st.info("👈 Sube un video para comenzar el análisis.")
    else:
        st.info("👈 Presiona 'Iniciar / Reiniciar Análisis' para comenzar.")
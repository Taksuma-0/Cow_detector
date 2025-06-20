import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np

## Iniciamos pagina
st.set_page_config(page_title="Dashboard de Detección de Vacas", page_icon="🐄", layout="wide")

##Modelos
MODELS = {
    "ResNet50 (Clasificación)": "C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/models/best_cow_model.pth",
    "YOLOv8 (Detección)": "C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/models/yolo_detection_best.pt",
    "YOLOv8 (Segmentación)": "C:/Users/welin/OneDrive/Escritorio/cow_detector/pagina_resultados/models/yolo_segmentation_best.pt"
}
CLASS_NAMES_RESNET = ['todas_las_no_vacas', 'todas_las_vacas']


# Función para cargar y procesar con ResNet
def process_frame_resnet(frame, model, device):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES_RESNET[predicted_idx.item()]
    confidence_score = confidence.item()
    
    display_text = f"{predicted_class.replace('todas_las_', '')}: {confidence_score*100:.1f}%"
    color = (0, 255, 0) if predicted_class == 'todas_las_vacas' else (0, 0, 255)
    
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return frame, 1 if predicted_class == 'todas_las_vacas' else 0

# Función para procesar con YOLO
def process_frame_yolo(frame, model, confidence_threshold):
    # Hacer inferencia
    results = model.track(frame, conf=confidence_threshold,persist=True , verbose=False,iou=0.3)
    overlay = frame.copy()
    
    # Iterar sobre cada detección en los resultados
    for result in results[0]:
        # se genera mascara de Segmentación 
        # verificar si hay una máscara en la detección
        if result.masks is not None:
            # Tomar las coordenadas del polígono de la máscara
            polygon = result.masks.xy[0].astype(np.int32)
            # Dibujar el polígono relleno sobre la capa de superposición
            cv2.fillPoly(overlay, [polygon], color=(0, 0, 250)) # Color verde para la máscara

        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])  # Tomar las coordenadas de la caja
        conf = result.boxes.conf[0].item() #Tomar la confianza y el nombre de la clase
        cls_name = model.names[int(result.boxes.cls[0])]
        
        label = f"{cls_name}: {conf:.2f}"
        
        # Dibujar el rectángulo sobre el frame original
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        # Dibujar la etiqueta sobre el frame original
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    ## Esto crea el efecto de transparencia
    alpha = 0.4 ## nivel de transparencia 
    annotated_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    ## Contar las detecciones en tiempo rial
    cow_count = len(results[0])
    
    return annotated_frame, cow_count


st.title("🐄 Dashboard Comparativo de Detección de Vacas")

st.sidebar.header("Configuración")
selected_model_name = st.sidebar.selectbox("Selecciona un modelo", list(MODELS.keys()))
confidence_threshold = st.sidebar.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)
uploaded_file = st.sidebar.file_uploader("Carga tu video", type=['mp4', 'mov', 'avi'])

@st.cache_resource
def load_model(model_name):
    path = MODELS[model_name]
    if "YOLO" in model_name:
        model = YOLO(path)
    elif "ResNet" in model_name:
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES_RESNET))
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
    else:
        raise ValueError("Modelo no soportado")
    return model

model = load_model(selected_model_name)
if "ResNet" in selected_model_name:
    model.eval()


if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    video_placeholder = st.empty()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analíticas en Vivo")
    cow_count_placeholder = st.sidebar.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if "YOLO" in selected_model_name:
            annotated_frame, cow_count = process_frame_yolo(frame, model, confidence_threshold)
        elif "ResNet" in selected_model_name:
            # ResNet no cuenta, solo clasifica. Mostramos 'Presente' o 'Ausente'.
            annotated_frame, presence = process_frame_resnet(frame, model, torch.device('cpu'))
            cow_count = "Presente" if presence == 1 and confidence_threshold < 0.8 else "Ausente"
        
        cow_count_placeholder.metric("Conteo/Estado de Vacas", cow_count)
        video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    st.success("Procesamiento de video finalizado.")
else:
    st.info("Sube un video y selecciona un modelo para comenzar.")
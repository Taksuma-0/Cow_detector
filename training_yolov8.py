data_yaml_path = "C:/Users/welin/OneDrive/Escritorio/cow_detector/Data/data.yaml"


from ultralytics import YOLO

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---

# 1. Cargar el modelo YOLOv8 'small' pre-entrenado.
model = YOLO("yolov8s.pt") 

data_yaml_path = "C:/Users/welin/OneDrive/Escritorio/cow_detector/Data/data.yaml"

# 3. Hiperparámetros de entrenamiento optimizados.
num_epochs = 200        # Un número alto, Early Stopping se encargará de parar si es necesario.
patience_epochs = 30    # Parada temprana si no mejora en 30 épocas.
image_size = 640
batch_size = 16         # Batch size adecuado para el modelo 's' en tu GPU.

# --- INICIAR EL ENTRENAMIENTO ---
if __name__ == '__main__':
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=num_epochs,
            patience=patience_epochs,
            batch=batch_size,
            imgsz=image_size,
            
            # --- RECETA DE AUMENTACIÓN AGRESIVA Y ESPECIALIZADA ---
            augment=True, 
            
            # Transformaciones geométricas para simular variedad de ángulos y distancias.
            degrees=10.0,
            translate=0.1,
            scale=0.2,
            shear=0.1,
            perspective=0.001,
            fliplr=0.5,
            
            # Técnicas avanzadas para combatir la oclusión y mejorar la generalización.
            mixup=0.1,
            copy_paste=0.1 # Muy potente para enseñar al modelo sobre oclusión.
        )
        print("✅ ¡Entrenamiento definitivo completado exitosamente!")
        
    except Exception as e:
        print(f"🚨 Ocurrió un error durante el entrenamiento: {e}")
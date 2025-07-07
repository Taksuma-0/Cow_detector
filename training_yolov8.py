from ultralytics import YOLO

model = YOLO("yolov8s.pt") 

data_yaml_path = "C:/Users/welin/OneDrive/Escritorio/cow_detector/Data/data.yaml"


num_epochs = 200       
patience_epochs = 30    
image_size = 640
batch_size = 16         


if __name__ == '__main__':
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=num_epochs,
            patience=patience_epochs,
            batch=batch_size,
            imgsz=image_size,
            
           
            augment=True, 
            
            # Transformaciones geométricas para simular variedad de ángulos y distancias.
            degrees=10.0,
            translate=0.1,
            scale=0.2,
            shear=0.1,
            perspective=0.001,
            fliplr=0.5,
            
           
            mixup=0.1,
            copy_paste=0.1 
        )
        print("✅ ¡Entrenamiento definitivo completado exitosamente!")
        
    except Exception as e:
        print(f"Ocurrió un error durante el entrenamiento: {e}")

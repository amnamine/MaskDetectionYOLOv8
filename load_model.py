from ultralytics import YOLO

def load_model_and_print_classes():
    """
    Load the YOLOv8 model from 'mask.pt' and print the number of classes
    """
    try:
        # Load the YOLOv8 model
        model = YOLO('mask.pt')
        
        # Get the number of classes
        num_classes = len(model.names)
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {model.names}")
        
        # Print each class with its index
        print("\nClass details:")
        for idx, class_name in model.names.items():
            print(f"  Class {idx}: {class_name}")
            
        return model, num_classes
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

if __name__ == "__main__":
    model, num_classes = load_model_and_print_classes() 
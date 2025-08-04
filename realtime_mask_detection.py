import cv2
import numpy as np
from ultralytics import YOLO
import time

class RealtimeMaskDetection:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        self.load_model()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.model = YOLO('mask.pt')
            print("✅ Model loaded successfully!")
            print(f"📊 Confidence threshold: {self.confidence_threshold}")
            print(f"🎭 Classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
    
    def initialize_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            exit(1)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📹 Camera initialized successfully!")
    
    def get_class_color(self, class_id):
        """Get color for each class"""
        colors = {
            0: (0, 165, 255),    # Orange for mask_weared_incorrect
            1: (0, 255, 0),      # Green for with_mask
            2: (0, 0, 255)       # Red for without_mask
        }
        return colors.get(class_id, (255, 255, 255))
    
    def get_class_name(self, class_id):
        """Get class name"""
        names = {
            0: "Mask Incorrect",
            1: "With Mask",
            2: "No Mask"
        }
        return names.get(class_id, "Unknown")
    
    def draw_detections(self, frame, results):
        """Draw detection boxes and labels on frame"""
        detections_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get confidence
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Apply confidence threshold
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.get_class_name(cls)
                    color = self.get_class_color(cls)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get label size
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    detections_count += 1
        
        return detections_count
    
    def draw_stats(self, frame, fps, detections_count):
        """Draw statistics on frame"""
        # Create stats background
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Draw stats text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detections_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.confidence_threshold}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw legend
        legend_y = 150
        cv2.putText(frame, "Legend:", (20, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]
        names = ["With Mask", "Mask Incorrect", "No Mask"]
        
        for i, (color, name) in enumerate(zip(colors, names)):
            y_pos = legend_y + 25 + (i * 20)
            cv2.circle(frame, (20, y_pos - 5), 5, color, -1)
            cv2.putText(frame, name, (35, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop"""
        self.initialize_camera()
        
        print("\n🎯 Starting real-time mask detection...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'c' to change confidence threshold")
        print("   - Press 'h' to show/hide help")
        
        frame_count = 0
        start_time = time.time()
        fps = 0.0  # Initialize fps variable
        show_help = True
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Draw detections
            detections_count = self.draw_detections(frame, results)
            
            # Draw statistics
            self.draw_stats(frame, fps, detections_count)
            
            # Show help text
            if show_help:
                help_text = [
                    "Controls:",
                    "Q - Quit",
                    "S - Save Screenshot", 
                    "C - Change Confidence",
                    "H - Hide/Show Help"
                ]
                
                for i, text in enumerate(help_text):
                    y_pos = frame.shape[0] - 120 + (i * 20)
                    cv2.putText(frame, text, (frame.shape[1] - 200, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('🎭 Real-time Mask Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"mask_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('c'):
                self.change_confidence_threshold()
            elif key == ord('h'):
                show_help = not show_help
        
        self.cleanup()
    
    def change_confidence_threshold(self):
        """Change confidence threshold interactively"""
        print(f"\n🔧 Current confidence threshold: {self.confidence_threshold}")
        try:
            new_threshold = float(input("Enter new confidence threshold (0.1-1.0): "))
            if 0.1 <= new_threshold <= 1.0:
                self.confidence_threshold = new_threshold
                print(f"✅ Confidence threshold updated to: {self.confidence_threshold}")
            else:
                print("❌ Invalid threshold. Must be between 0.1 and 1.0")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")

def main():
    print("🎭 Real-time Mask Detection System")
    print("=" * 40)
    
    # Initialize detection system
    detector = RealtimeMaskDetection(confidence_threshold=0.5)
    
    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main() 
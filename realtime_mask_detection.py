import cv2
import numpy as np
from ultralytics import YOLO
import time

class RealtimeMaskDetection:
    def __init__(self, confidence_threshold=0.3):
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
        
        # Get screen dimensions - user's actual screen resolution
        screen_width = 1366  # User's screen width
        screen_height = 768  # User's screen height
        
        # Set camera properties for full screen
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📹 Camera initialized successfully!")
        print(f"🖥️ Full screen resolution: {screen_width}x{screen_height}")
    
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
                    
                    # Ensure bounding box is within image bounds
                    height, width = frame.shape[:2]
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    # Get class
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.get_class_name(cls)
                    color = self.get_class_color(cls)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get label size with smaller font
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                    
                    # Ensure label background is within image bounds
                    label_x = x1
                    label_y = max(label_size[1] + 3, y1 - 3)  # Ensure label doesn't go above image
                    
                    # Draw label background
                    cv2.rectangle(frame, (label_x, label_y - label_size[1]), 
                                (label_x + label_size[0], label_y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (label_x, label_y - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    detections_count += 1
        
        return detections_count
    
    def draw_stats(self, frame, detections_count):
        """Draw statistics on frame"""
        # Create smaller stats background to maximize image space
        cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), 1)
        
        # Draw stats text with smaller font
        cv2.putText(frame, f"Detections: {detections_count}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Confidence: {self.confidence_threshold}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw compact legend
        legend_y = 100
        cv2.putText(frame, "Legend:", (15, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]
        names = ["With Mask", "Mask Incorrect", "No Mask"]
        
        for i, (color, name) in enumerate(zip(colors, names)):
            y_pos = legend_y + 20 + (i * 15)
            cv2.circle(frame, (15, y_pos - 3), 3, color, -1)
            cv2.putText(frame, name, (25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def run_detection(self):
        """Main detection loop"""
        self.initialize_camera()
        
        print("\n🎯 Starting real-time mask detection...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'h' to show/hide help")
        
        show_help = True
        window_created = False
        
        # Destroy any existing windows
        cv2.destroyAllWindows()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Draw detections
            detections_count = self.draw_detections(frame, results)
            
            # Draw statistics
            self.draw_stats(frame, detections_count)
            
            # Show help text
            if show_help:
                help_text = [
                    "Controls:",
                    "Q - Quit",
                    "S - Save Screenshot", 
                    "H - Hide/Show Help"
                ]
                
                for i, text in enumerate(help_text):
                    y_pos = frame.shape[0] - 80 + (i * 15)
                    cv2.putText(frame, text, (frame.shape[1] - 150, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Create window only once on first frame
            if not window_created:
                cv2.namedWindow('🎭 Real-time Mask Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('🎭 Real-time Mask Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                window_created = True
            
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
            elif key == ord('h'):
                show_help = not show_help
        
        self.cleanup()
    
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
    detector = RealtimeMaskDetection(confidence_threshold=0.3)
    
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
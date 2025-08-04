import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class MaskDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Detection System - YOLOv8")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.original_image = None
        self.image_path = None
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.model = YOLO('mask.pt')
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🎭 Mask Detection System",
            font=("Helvetica", 24, "bold"),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = tk.Label(
            main_frame,
            text="Powered by YOLOv8 - Detect mask usage in images",
            font=("Helvetica", 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(pady=(0, 20))
        
        # Load Image Button
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=("Helvetica", 12, "bold"),
            bg='#3498db',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        # Predict Button
        self.predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict",
            command=self.predict_image,
            font=("Helvetica", 12, "bold"),
            bg='#27ae60',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Reset Button
        self.reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset",
            command=self.reset_image,
            font=("Helvetica", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Image display frame
        image_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Image label
        self.image_label = tk.Label(
            image_frame,
            text="📸 Click 'Load Image' to start",
            font=("Helvetica", 16),
            fg='#bdc3c7',
            bg='#34495e',
            width=50,
            height=15
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg='#2c3e50')
        results_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Results label
        self.results_label = tk.Label(
            results_frame,
            text="Ready to detect masks!",
            font=("Helvetica", 12),
            fg='#ecf0f1',
            bg='#2c3e50',
            wraplength=800
        )
        self.results_label.pack()
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Status: Ready",
            font=("Helvetica", 10),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        self.status_label.pack(pady=(10, 0))
        
        # Bind hover effects
        self.bind_hover_effects()
    
    def bind_hover_effects(self):
        """Add hover effects to buttons"""
        def on_enter(event):
            event.widget.configure(bg='#2980b9' if event.widget == self.load_btn else 
                                 '#229954' if event.widget == self.predict_btn else '#c0392b')
        
        def on_leave(event):
            event.widget.configure(bg='#3498db' if event.widget == self.load_btn else 
                                 '#27ae60' if event.widget == self.predict_btn else '#e74c3c')
        
        for btn in [self.load_btn, self.predict_btn, self.reset_btn]:
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
    
    def load_image(self):
        """Load an image from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                
                # Resize image for display
                display_image = self.resize_image_for_display(self.original_image)
                self.current_image = display_image
                
                # Convert to PhotoImage
                image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update image label
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Enable buttons
                self.predict_btn.configure(state=tk.NORMAL)
                self.reset_btn.configure(state=tk.NORMAL)
                
                # Update status
                self.status_label.configure(text=f"Status: Image loaded - {os.path.basename(file_path)}")
                self.results_label.configure(text="Image loaded successfully! Click 'Predict' to detect masks.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def resize_image_for_display(self, image, max_width=600, max_height=400):
        """Resize image for display while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        
        return image
    
    def predict_image(self):
        """Run prediction on the loaded image"""
        if self.original_image is None or self.model is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        try:
            self.status_label.configure(text="Status: Running prediction...")
            self.root.update()
            
            # Run prediction
            results = self.model(self.original_image)
            
            # Process results
            result_image = self.original_image.copy()
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        
                        # Choose color based on class
                        if cls == 0:  # mask_weared_incorrect
                            color = (0, 165, 255)  # Orange
                        elif cls == 1:  # with_mask
                            color = (0, 255, 0)    # Green
                        else:  # without_mask
                            color = (0, 0, 255)    # Red
                        
                        # Draw bounding box
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(result_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'color': color
                        })
            
            # Update display
            display_result = self.resize_image_for_display(result_image)
            image_rgb = cv2.cvtColor(display_result, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Update results
            if detections:
                result_text = f"Detection Results ({len(detections)} objects found):\n"
                for i, det in enumerate(detections, 1):
                    result_text += f"{i}. {det['class']} (Confidence: {det['confidence']:.2f})\n"
            else:
                result_text = "No masks detected in the image."
            
            self.results_label.configure(text=result_text)
            self.status_label.configure(text="Status: Prediction completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_label.configure(text="Status: Prediction failed")
    
    def reset_image(self):
        """Reset the image to original state"""
        # Reset to initial state - completely clear everything
        self.image_label.configure(image="", text="📸 Click 'Load Image' to start")
        self.predict_btn.configure(state=tk.DISABLED)
        self.reset_btn.configure(state=tk.DISABLED)
        self.results_label.configure(text="Ready to detect masks!")
        self.status_label.configure(text="Status: Ready")
        
        # Clear all image variables
        self.current_image = None
        self.original_image = None
        self.image_path = None

def main():
    root = tk.Tk()
    app = MaskDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
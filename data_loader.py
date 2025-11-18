
import os
import cv2
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, input_folder):
        self.input_folder = Path(input_folder)
        self.image_extensions = ['.jpg', '.jpeg']
    
    def get_image_files(self):
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.input_folder.glob(f'*{ext}'))
            image_files.extend(self.input_folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def load_all_images(self):
    
        image_files = self.get_image_files()
        images = {}
        
        for img_path in image_files:
            image = self.load_image(img_path)
            if image is not None:
                images[img_path.name] = image
            else:
                print(f"Warning: Could not load {img_path.name}")
        
        return images
    
    def get_image_count(self):
        return len(self.get_image_files())


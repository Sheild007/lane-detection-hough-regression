import numpy as np
import cv2

class ColorFilter:
    def __init__(self):
        # Yellow:
        self.yellow_lower = np.array([15, 50, 80])
        self.yellow_upper = np.array([40, 255, 255])

        # White:
        self.white_lower = np.array([0, 0, 100])
        self.white_upper = np.array([180, 255, 255])
    
    def rgb_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    def filter_binary_image(self, binary_image, hsv_image):
        mask_yellow = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
        mask_white = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        
        binary_output = (combined_mask > 0).astype(np.uint8)
        return cv2.bitwise_and(binary_image, binary_output)
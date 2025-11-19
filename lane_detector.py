import numpy as np
import cv2
from lane_detection.hough import HoughTransform
from lane_detection.regression import LinearRegression
from lane_detection.color_filter import ColorFilter
from lane_detection.canny import canny_edge_detector, apply_gaussian_filter

class LaneDetector:
    def __init__(self, canny_low=15, canny_high=50):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.color_filter = ColorFilter()
        self.hough_transform = HoughTransform(threshold=20, min_line_length=15)
        self.linear_regression = LinearRegression()

    def manual_dilate(self, image, kernel_size=3):
       
        pad = kernel_size // 2
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
        output = np.zeros_like(image)
    
        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted = padded[i:i+image.shape[0], j:j+image.shape[1]]
                output = np.maximum(output, shifted)
        return output

    def apply_contrast_enhancement(self, gray_image):
       
        hist, bins = np.histogram(gray_image.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0) 
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
   
        return cdf[gray_image]

    def define_region_of_interest(self, image):
      
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x_bl, y_bl = int(w * 0.05), h
        x_tl, y_tl = int(w * 0.46), int(h * 0.62)
        x_tr, y_tr = int(w * 0.54), int(h * 0.62)
        x_br, y_br = int(w * 0.95), h
      
        Y, X = np.indices((h, w))
        left_boundary = np.interp(Y, [y_tl, y_bl], [x_tl, x_bl])
        right_boundary = np.interp(Y, [y_tr, y_br], [x_tr, x_br])
        
        roi_condition = (Y >= y_tl) & (X >= left_boundary) & (X <= right_boundary)
        
        mask[roi_condition] = 255
        return mask, None 

    
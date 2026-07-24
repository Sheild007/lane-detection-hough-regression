import numpy as np
import cv2
from lane_detection.hough import HoughTransform
from lane_detection.regression import LinearRegression
from lane_detection.color_filter import ColorFilter
from lane_detection.canny import canny_edge_detector

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

    def apply_roi_mask(self, image, roi_mask):
    
        masked_image = image.copy()
        masked_image[roi_mask == 0] = 0
        return masked_image
    
    def filter_lines_by_slope(self, lines, image_width):
        left_lines = []
        right_lines = []
        if lines is None: 
            return [], []
        center_x = image_width / 2
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 - x1 == 0:
                continue 
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3: 
                continue
            line_center = (x1 + x2) / 2
            if slope < 0 and line_center < center_x: 
                left_lines.append(line)
            elif slope > 0 and line_center > center_x: 
                right_lines.append(line)
        return left_lines, right_lines
    
    def fit_lane_lines(self, left_lines, right_lines, image_shape):
        h, w = image_shape[:2]
        y_bottom = h
        y_top = int(h * 0.65)
        def get_points(lines):
            pts = []
            for x1, y1, x2, y2 in lines:
                pts.append([x1, y1])
                pts.append([x2, y2])
            return pts
        left_line, right_line = None, None
        left_pts = get_points(left_lines)
        if len(left_pts) > 0:
            slope, intercept = self.linear_regression.fit(left_pts)
            if slope is not None and abs(slope) > 0.2: 
                left_line = self.linear_regression.get_line_points(slope, intercept, y_bottom, y_top)
        right_pts = get_points(right_lines)
        if len(right_pts) > 0:
            slope, intercept = self.linear_regression.fit(right_pts)
            if slope is not None and abs(slope) > 0.2:
                right_line = self.linear_regression.get_line_points(slope, intercept, y_bottom, y_top)
        return left_line, right_line
    
    def draw_lane_lines(self, image, left_line, right_line):
        result = image.copy()
        if left_line:
            cv2.line(result, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 10)
        if right_line:
            cv2.line(result, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 10)
        return result
    
    def process_image(self, image, save_intermediate=False):
        results = {}
        h, w = image.shape[:2]
        
        # 1. HSV
        hsv_image = self.color_filter.rgb_to_hsv(image) # Allowed built-in
        results['hsv'] = hsv_image 

        # 2. Color Filter 
        binary_B = np.ones((h, w), dtype=np.uint8)
        color_mask = self.color_filter.filter_binary_image(binary_B, hsv_image)
        dilated_mask = self.manual_dilate(color_mask, kernel_size=3)
        results['color_filtered'] = dilated_mask * 255

        # 3. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Allowed built-in
        gray_enhanced = self.apply_contrast_enhancement(gray) # MANUAL EQ
        results['grayscale'] = gray_enhanced
        
        # 4. Canny
        canny_edges = canny_edge_detector(
            gray_enhanced, 
            binary_mask=dilated_mask, 
            low_threshold=self.canny_low, 
            high_threshold=self.canny_high
        )
        results['canny_edges'] = canny_edges

        # 5. ROI 
        roi_mask, _ = self.define_region_of_interest(image)
        roi_edges = self.apply_roi_mask(canny_edges, roi_mask)
        results['roi_mask'] = roi_mask
        results['roi_edges'] = roi_edges

        # 6. Hough
        lines = self.hough_transform.transform(roi_edges)
        hough_viz = image.copy()
        if lines:
            for l in lines:
                cv2.line(hough_viz, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
        results['hough_lines_viz'] = hough_viz
        results['hough_data'] = lines

        # 7. Filter & Regression
        left_lines, right_lines = self.filter_lines_by_slope(lines, w)
        left_fit, right_fit = self.fit_lane_lines(left_lines, right_lines, image.shape)
        results['final'] = self.draw_lane_lines(image, left_fit, right_fit)
        
        return results
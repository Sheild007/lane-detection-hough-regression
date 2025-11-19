import numpy as np

class LinearRegression:
    
    @staticmethod
    def fit(points):
        if len(points) == 0:
            return None, None
        
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        
        n = len(points)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        
        if abs(denominator) < 1e-10:  # Vertical line case
            slope = 1e9 
            intercept = np.mean(y) - slope * np.mean(x)
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    @staticmethod
    def get_line_points(slope, intercept, y_start, y_end):
        if slope is None:
            return None
       
        if abs(slope) < 1e-5: 
            return None 
            
        try:
            x1 = int((y_start - intercept) / slope)
            x2 = int((y_end - intercept) / slope)
            return (x1, y_start, x2, y_end)
        except (OverflowError, ValueError):
            return None

import numpy as np

class HoughTransform:
 
    def __init__(self, rho_resolution=1, theta_resolution=np.pi/180, threshold=80,
                 min_line_length=50, max_line_gap=10):
     
        self.rho_resolution = rho_resolution
        self.theta_resolution = theta_resolution
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def transform(self, edge_image):
     
        h, w = edge_image.shape
        max_rho = int(np.sqrt(h**2 + w**2))
        num_rhos = 2 * max_rho
        num_thetas = int(np.pi / self.theta_resolution)
        
       
        accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
        edge_points = np.argwhere(edge_image > 0)
        
     
        for point in edge_points:
            y, x = point
            for theta_idx in range(num_thetas):
                theta = theta_idx * self.theta_resolution - np.pi / 2
                # Calculate rho ρ = x*cos(θ) + y*sin(θ)
                rho = x * np.cos(theta) + y * np.sin(theta)
                rho_idx = int(rho / self.rho_resolution + max_rho)
                
                if 0 <= rho_idx < num_rhos:
                    accumulator[rho_idx, theta_idx] += 1
    
        lines = []
        for rho_idx in range(num_rhos):
            for theta_idx in range(num_thetas):
                if accumulator[rho_idx, theta_idx] >= self.threshold:
                    rho = (rho_idx - max_rho) * self.rho_resolution
                    theta = theta_idx * self.theta_resolution - np.pi / 2
                    
                
                    line = self._rho_theta_to_line(rho, theta, h, w)
                    
                    if line is not None:
                        x1, y1, x2, y2 = line
                        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if line_length >= self.min_line_length:
                            lines.append(line)
        
        return lines
    

    def _rho_theta_to_line(self, rho, theta, h, w):
        
        points = []
        
        if abs(np.sin(theta)) > 0.003:
            y = int(rho / np.sin(theta))
            if 0 <= y < h:
                points.append((0, y))
      
        if abs(np.sin(theta)) > 0.003:
            y = int((rho - (w-1) * np.cos(theta)) / np.sin(theta))
            if 0 <= y < h:
                points.append((w-1, y))
       
        if abs(np.cos(theta)) > 0.003:
            x = int(rho / np.cos(theta))
            if 0 <= x < w:
                points.append((x, 0))

    
        if abs(np.cos(theta)) > 0.003:
            x = int((rho - (h-1) * np.sin(theta)) / np.cos(theta))
            if 0 <= x < w:
                points.append((x, h-1))

        unique_points = sorted(list(set(points)))
        
        if len(unique_points) >= 2:
            return (unique_points[0][0], unique_points[0][1], 
                    unique_points[-1][0], unique_points[-1][1])
        
        return None
    
  
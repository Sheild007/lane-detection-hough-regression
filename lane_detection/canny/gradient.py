import numpy as np

class Gradient:
    def __init__(self, fx, fy, scale_factor=1):
        self.fx = fx
        self.fy = fy
        self.scale_factor = scale_factor

    def compute_magnitude(self):
        
        mag = np.sqrt(self.fx**2 + self.fy**2) / self.scale_factor
        mag = np.clip(mag, 0, 255)
        
        return mag.astype(np.uint8)

    def compute_direction(self):
        direction = np.arctan2(self.fy, self.fx) * 180 / np.pi
        direction = (direction + 360) % 360
        return direction
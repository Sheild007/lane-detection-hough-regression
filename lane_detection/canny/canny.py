
from .convolution import Convolver
from .mask import GaussianMask
from .gradient import Gradient
from .nonmax import NonMaxSuppresser
from .hysteresis import Hysteresis
import numpy as np

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
  
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
 
    kernel = kernel / np.sum(kernel)
    
    convolver = Convolver(kernel)
    
    
    if len(image.shape) == 3:
        
        channels = []
        for c in range(3):
            channels.append(convolver.convolve(image[:, :, c]))
        filtered = np.stack(channels, axis=2)
    else:
       
        filtered = convolver.convolve(image)
        
    return filtered.astype(np.uint8)


def canny_edge_detector(grayscale_image, binary_mask, low_threshold=50, high_threshold=150, sigma=1.0, T=0.01, scale_factor=255):
   
    
    img = grayscale_image.astype(np.float32)
    
    gaussian_mask = GaussianMask(sigma=sigma, T=T)
    Gx, Gy, scale_factor = gaussian_mask.calculate_gradient(scale_factor=scale_factor)
    

    convolver_x = Convolver(Gx)
    convolver_y = Convolver(Gy)
    
    fx = convolver_x.convolve(img)
    fy = convolver_y.convolve(img)
    
  
    gradient = Gradient(fx, fy, scale_factor=scale_factor)
    magnitude = gradient.compute_magnitude()
    direction = gradient.compute_direction()
    
   
    nonmax_suppressor = NonMaxSuppresser(magnitude, direction)
    suppressed = nonmax_suppressor.suppress(magnitude, direction)
  
    hysteresis = Hysteresis(Th=high_threshold, Tl=low_threshold)
    final_edges = hysteresis.thresholding(suppressed)
    
    filtered_edges = final_edges.copy()
    filtered_edges[binary_mask == 0] = 0
    
    return filtered_edges

from convolution import Convolver
from mask import GaussianMask
from utils import apply_gaussian_filter
from gradient import Gradient
from nonmax import NonMaxSuppresser
from hysteresis import Hysteresis
import numpy as np

def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):


    gaussian_mask = GaussianMask(kernel_size=kernel_size, sigma=sigma)
    kernel = gaussian_mask.get_kernel()
    filtered = Convolver.convolve(image, kernel)
    return filtered.astype(np.uint8)


def canny_edge_detector(grayscale_image, binary_mask, low_threshold=50, high_threshold=150):
 
    smoothed = apply_gaussian_filter(grayscale_image)
    
    gradient = Gradient()
    magnitude, angle = gradient.compute(smoothed)
    
    non_max_suppresser = NonMaxSuppresser()
    suppressed = non_max_suppresser.suppress(magnitude, angle)
    
    hysteresis = Hysteresis(low_threshold=low_threshold, high_threshold=high_threshold)
    edges = hysteresis.apply(suppressed)
    
    filtered_edges = edges.copy()
    filtered_edges[binary_mask == 0] = 0
    
    return filtered_edges
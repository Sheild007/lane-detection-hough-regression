import numpy as np

def rgb_to_grayscale(image):
    if len(image.shape) == 3:
        grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return grayscale.astype(np.uint8)
    else:
        return image




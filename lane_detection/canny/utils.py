import numpy as np

def save_image(image_array, path):
    from PIL import Image
    img = Image.fromarray(image_array)
    img.save(path)
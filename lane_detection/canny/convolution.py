import numpy as np

class Convolver:
    def __init__(self, mask):
        self.mask = mask
        self.mH, self.mW = mask.shape
        self.padH, self.padW = self.mH // 2, self.mW // 2

    def convolve(self, image):
        H, W = image.shape
        padded_image = np.pad(image, ((self.padH, self.padH), (self.padW, self.padW)), 'constant')
        output = np.zeros_like(image, dtype=np.float32)

        for i in range(H):
            for j in range(W):
                region = padded_image[i:i+self.mH, j:j+self.mW]
                new_pixel = np.sum(region * self.mask)
                output[i, j] = new_pixel
        
        return output.astype(np.float32)
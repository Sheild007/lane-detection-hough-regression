import numpy as np

class GaussianMask:
    def __init__(self, sigma, T):
        self.sigma = sigma
        self.T = T
        self.filter_size, self.sHalf = self.calculate_filter_size()

    def calculate_filter_size(self):
        sHalf = round(np.sqrt(-np.log(self.T) * 2 * (self.sigma**2)))
        N = 2 * sHalf + 1
        return N, sHalf

    def calculate_gradient(self, scale_factor=255):
        sHalf = (self.filter_size - 1) // 2
        sequence = np.arange(-sHalf, sHalf + 1)
        X, Y = np.meshgrid(sequence, sequence)
        denominator = 2 * self.sigma ** 2
        numerator = X ** 2 + Y ** 2
        G = np.exp(-numerator / denominator)

        Gx = (X * G) / (-self.sigma ** 2)
        Gy = (Y * G) / (-self.sigma ** 2)

        Gx = np.round(Gx * scale_factor)
        Gy = np.round(Gy * scale_factor)

        return Gx, Gy, scale_factor

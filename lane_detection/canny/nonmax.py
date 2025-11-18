import numpy as np

class NonMaxSuppresser:

    def __init__(self, mag, dir):
        self.mag = mag
        self.dir = dir
    
    def quantize_direction(self, direction):
        normalized = direction % 180
        quantized = ((normalized + 22.5) / 45).astype(int) % 4
        
        return quantized.astype(np.uint8)

    def suppress(self, mag, dir):
       
       rows, cols = mag.shape
       quantized_dir = self.quantize_direction(dir)
       suppressed = np.zeros_like(mag)

       for r in range(1, rows - 1):
           for c in range(1, cols - 1):
                direction = quantized_dir[r, c]
                current_mag = mag[r, c]

                if direction == 0:  # Horizontal gradient -> Vertical edge
                    # Compare with East and West neighbors
                    neighbor1 = mag[r, c - 1]
                    neighbor2 = mag[r, c + 1]
                elif direction == 1:  # Diagonal gradient (NE-SW) -> NW-SE edge
                    # Compare with NW and SE neighbors
                    neighbor1 = mag[r - 1, c - 1]
                    neighbor2 = mag[r + 1, c + 1]
                elif direction == 2:  # Vertical gradient -> Horizontal edge
                    # Compare with North and South neighbors
                    neighbor1 = mag[r - 1, c]
                    neighbor2 = mag[r + 1, c]
                else:  # direction == 3: Diagonal gradient (NW-SE) -> NE-SW edge
                    # Compare with NE and SW neighbors
                    neighbor1 = mag[r - 1, c + 1]
                    neighbor2 = mag[r + 1, c - 1]

                if current_mag >= neighbor1 and current_mag >= neighbor2:
                    suppressed[r, c] = current_mag
                else:
                    suppressed[r, c] = 0
        
       return suppressed
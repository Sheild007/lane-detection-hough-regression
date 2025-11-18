import numpy as np 


class Hysteresis:

    def __init__(self, Th, Tl):
        self.Th = Th
        self.Tl = Tl
    
        self.Neighbours = [ 
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    def flood_fill(self, img, start_i, start_j, visited):
        h, w = img.shape
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            
            if i < 0 or i >= h or j < 0 or j >= w:
                continue
            if visited[i, j]:
                continue
            if img[i, j] < self.Tl:
                continue
                
      
            visited[i, j] = True
            img[i, j] = 255
            
            
            for dx, dy in self.Neighbours:
                ni, nj = i + dx, j + dy
                if 0 <= ni < h and 0 <= nj < w:
                    if not visited[ni, nj] and img[ni, nj] >= self.Tl:
                        stack.append((ni, nj))

    def thresholding(self, img):
      
        result = img.copy()
        h, w = result.shape
        visited = np.zeros_like(result, dtype=bool)
        
        for i in range(h):
            for j in range(w):
                if result[i, j] >= self.Th and not visited[i, j]:
                    self.flood_fill(result, i, j, visited)
                elif result[i, j] < self.Tl:
                    result[i, j] = 0
    
        for i in range(h):
            for j in range(w):
                if result[i, j] != 255:
                    result[i, j] = 0
        
        return result

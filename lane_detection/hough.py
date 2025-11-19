import numpy as np

class HoughTransform:
    def __init__(self, rho_resolution=1, theta_resolution=np.pi/180, threshold=80,
                 min_line_length=30, max_line_gap=10):
        self.rho_res = rho_resolution
        self.theta_res = theta_resolution
        self.threshold = threshold
        self.min_line_length = min_line_length

    def transform(self, edge_image):
        h, w = edge_image.shape
        # 1. Setup Accumulator
        diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
        rhos = np.arange(-diag_len, diag_len + 1, self.rho_res)
        thetas = np.arange(0, np.pi, self.theta_res)
        
        # Cache cos/sin for speed
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)
        num_rhos = len(rhos)

        # 2. Get Edge Coordinates
        y_idxs, x_idxs = np.nonzero(edge_image)
        
        # 3. Voting (Vectorized)
        accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint64)
        

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            
            # rho = x*cos + y*sin
            edge_rhos = x * cos_t + y * sin_t
            
            # Map rho to index
            rho_idxs = np.round(edge_rhos + diag_len).astype(np.int32)
            
            # Increment accumulator
            valid_mask = (rho_idxs >= 0) & (rho_idxs < num_rhos)
            accumulator[rho_idxs[valid_mask], np.where(valid_mask)[0]] += 1

        # 4. Find Lines > Threshold
        r_idxs, t_idxs = np.where(accumulator > self.threshold)
        
        # Sort by vote count (descending) to process strongest lines first
        votes = accumulator[r_idxs, t_idxs]
        sort_idx = np.argsort(votes)[::-1]
        r_idxs = r_idxs[sort_idx]
        t_idxs = t_idxs[sort_idx]
        
        # Limit to top 100 lines to prevent noise explosion
        if len(r_idxs) > 100:
            r_idxs = r_idxs[:100]
            t_idxs = t_idxs[:100]

        lines = []
        for i in range(len(r_idxs)):
            rho = r_idxs[i] - diag_len
            theta = thetas[t_idxs[i]]
            
            line = self._rho_theta_to_line(rho, theta, h, w)
            if line is not None:
                lines.append(line)
                
        return lines

    def _rho_theta_to_line(self, rho, theta, h, w):
    
        points = []
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        # Intersection with vertical borders (x=0, x=w-1)
        if abs(sin_t) > 1e-3:
            # x=0
            y0 = int(rho / sin_t)
            if 0 <= y0 < h: points.append((0, y0))
            # x=w-1
            y1 = int((rho - (w-1)*cos_t) / sin_t)
            if 0 <= y1 < h: points.append((w-1, y1))
            
        # Intersection with horizontal borders (y=0, y=h-1)
        if abs(cos_t) > 1e-3:
            # y=0
            x0 = int(rho / cos_t)
            if 0 <= x0 < w: points.append((x0, 0))
            # y=h-1
            x1 = int((rho - (h-1)*sin_t) / cos_t)
            if 0 <= x1 < w: points.append((x1, h-1))
            
        points = sorted(list(set(points)))
        
        if len(points) >= 2:
           
            return (points[0][0], points[0][1], points[-1][0], points[-1][1])
        return None
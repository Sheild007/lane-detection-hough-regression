import numpy as np
import cv2
from lane_detection.hough import HoughTransform
from lane_detection.regression import LinearRegression
from lane_detection.color_filter import ColorFilter
from lane_detection.canny import canny_edge_detector, apply_gaussian_filter

class LaneDetector:
    def __init__(self, canny_low=15, canny_high=50):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.color_filter = ColorFilter()
        self.hough_transform = HoughTransform(threshold=20, min_line_length=15)
        self.linear_regression = LinearRegression()

   
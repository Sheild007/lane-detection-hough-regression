# Roll Number: BSCS22008
# Name: Muhammad Usman Muneer
# Assignment Number: 04

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from lane_detector import LaneDetector
from data_loader import DataLoader

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
   
    subfolders = [
        'step1_hsv', 
        'step2_gaussian', 
        'step3_color_mask', 
        'step4_grayscale_enhanced', 
        'step5_canny_edges', 
        'step6_roi_mask', 
        'step7_roi_edges', 
        'step8_hough_lines', 
        'step9_final_output'
    ]
    
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
    
    data_loader = DataLoader(input_folder)
    image_files = data_loader.get_image_files()
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return


    detector = LaneDetector(canny_low=15, canny_high=50)
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        image_rgb = data_loader.load_image(img_path)
        if image_rgb is None: continue
        
        base_name = Path(img_path).stem
        print(f"  - {base_name}...", end="", flush=True)
        
       
        results = detector.process_image(image_rgb, save_intermediate=True)
    
        if 'hsv' in results:
            cv2.imwrite(os.path.join(output_folder, 'step1_hsv', f'{base_name}_hsv.jpg'), 
                       results['hsv'])

        
        if 'gaussian_filtered' in results:
            blur_bgr = cv2.cvtColor(results['gaussian_filtered'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder, 'step2_gaussian', f'{base_name}_blur.jpg'), 
                       blur_bgr)

        if 'color_filtered' in results:
            cv2.imwrite(os.path.join(output_folder, 'step3_color_mask', f'{base_name}_mask.jpg'), 
                       results['color_filtered'])

        
        if 'grayscale' in results:
            cv2.imwrite(os.path.join(output_folder, 'step4_grayscale_enhanced', f'{base_name}_gray.jpg'), 
                       results['grayscale'])

       
        if 'canny_edges' in results:
            cv2.imwrite(os.path.join(output_folder, 'step5_canny_edges', f'{base_name}_canny.jpg'), 
                       results['canny_edges'])

    
        if 'roi_mask' in results:
            cv2.imwrite(os.path.join(output_folder, 'step6_roi_mask', f'{base_name}_roi_mask.jpg'), 
                       results['roi_mask'])

        if 'roi_edges' in results:
            cv2.imwrite(os.path.join(output_folder, 'step7_roi_edges', f'{base_name}_roi_edges.jpg'), 
                       results['roi_edges'])

        
        if 'hough_lines_viz' in results:
            hough_bgr = cv2.cvtColor(results['hough_lines_viz'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder, 'step8_hough_lines', f'{base_name}_hough.jpg'), 
                       hough_bgr)
        
      
        if 'final' in results:
            final_bgr = cv2.cvtColor(results['final'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder, 'step9_final_output', f'{base_name}_final.jpg'), 
                       final_bgr)

            cv2.imwrite(os.path.join(output_folder, f'{base_name}_final.jpg'), final_bgr)
        
        print(" Done.")

def main():
    parser = argparse.ArgumentParser(description="Lane Detection Pipeline")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to input images")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to save results")
    args = parser.parse_args()
    
    process_images(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main() 
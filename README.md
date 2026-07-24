# Lane Detection — Hough Transform + Linear Regression

Detects road lane lines in images using a pipeline that goes from color filtering to Hough voting to a final fitted lane. Most of the core operations are manual — the Hough transform, the Canny detector, and the regression are not library calls.

## Pipeline

1. **Color filtering** (HSV) — isolates white and yellow lane markings
2. **Contrast enhancement** — histogram equalization on grayscale channel (manual, not `cv2.equalizeHist`)
3. **Edge detection** — custom Canny (same implementation as [canny-edge-detector-from-scratch](https://github.com/Sheild007/canny-edge-detector-from-scratch))
4. **Region of interest** — trapezoid mask focused on the lower half of the frame
5. **Hough transform** — votes in (ρ, θ) space to extract line candidates
6. **Slope filtering** — separates left/right lines by sign of slope and position relative to image center
7. **Linear regression** — fits a single representative line per side and extends it to full lane height

## How to run

```bash
pip install numpy opencv-python
python main.py --input_folder ./images --output_folder ./results
```

Results include intermediate steps (HSV mask, edges, ROI, raw Hough lines, final overlay) saved per image.

## What I'd improve

The ROI is hardcoded as a fraction of image dimensions — it doesn't adapt to camera angle or road curvature. A perspective warp (bird's-eye view) before Hough would make the regression more stable on curved roads.

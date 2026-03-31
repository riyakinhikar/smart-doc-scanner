# Smart Document Scanner \& Analyzer

A command line Computer Vision application that includes document detection in photographs, document extraction with perspective correction, and extensive visual analysis in the form of feature extraction and image segmentation. 

This project is a real-world application encompassing the key Computer Vision concepts in CSE3010 such as image preprocessing, edge detection, projective transformations, feature extraction (SIFT, ORB, HOG, Harris), and segmentation (K-Means, Watershed, Region Growing).

\---

## Table of Contents

* [Problem Statement](#problem-statement)
* [Project Architecture](#project-architecture)
* [Setup Instructions](#setup-instructions)
* [How to Run](#how-to-run)
* [Pipeline Description](#pipeline-description)
* [Course Concepts Applied](#course-concepts-applied)
* [Output Examples](#output-examples)
* [Project Structure](#project-structure)

\---

## Problem Statement

Each day hundreds of millions of people shoot documents – receipts, notes, certificates, whiteboards – with their smartphones. The photos are usually perspective-distorted, poorly-lit, and contain noise in the background. Existing scanner apps fix this issue, but to understand how they work, one has to build the underlying Computer Vision pipeline from scratch. 

This project develops a full document scanning and analysis tool from scratch using classical (no deep learning) CV techniques, proving that fundamental algorithms studied in this course can address a practical, everyday problem. ---

## Project Architecture

```
Input Image → Preprocessing → Edge Detection → Contour Detection → Perspective Transform → Clean Scan
                                                                          ↓
                                                              Feature Extraction (SIFT, ORB, HOG, Harris)
                                                                          ↓
                                                              Segmentation (K-Means, Watershed, Text Detection)
```

\---

## Setup Instructions

### Prerequisites

You need Python 3.8 or higher and pip installed on your system. This project runs entirely from the command line and does not require a GUI.

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/smart-doc-scanner.git
cd smart-doc-scanner
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\\Scripts\\activate           # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs NumPy and OpenCV (including the contrib modules for SIFT).

### Verify Installation

```bash
python -c "import cv2; print(f'OpenCV version: {cv2.\_\_version\_\_}')"
```

You should see the OpenCV version printed without errors.

\---

## How to Run

The application has three commands: `demo`, `scan`, and `analyze`.

### Run the Full Demo (No Input Image Needed)

This generates a synthetic document image and runs the complete pipeline on it. This is the best way to verify everything works.

```bash
python main.py demo
```

Outputs will be saved to the `output/` directory.

### Scan a Document from a Photo

```bash
python main.py scan path/to/your/photo.jpg
```

To specify a custom output directory:

```bash
python main.py scan photo.jpg -o my\_results/
```

### Analyze Features and Segments

```bash
python main.py analyze path/to/image.jpg
```

This runs feature extraction (ORB, SIFT, HOG, Harris corners) and segmentation (K-Means, Watershed, text region detection) on the input image.

### Quick Test with Built-in Sample

```bash
python main.py scan demo
python main.py analyze demo
```

Both commands accept `demo` as the image argument to use the built-in synthetic sample.

\---

## Pipeline Description

### Phase 1: Document Scanning

Step 1 — Preprocessing (Module 1) 



The input image is converted into grayscale and suppressed by Gaussian blur. This removes any high-frequency noise that would have otherwise produced spurious edges in the next step. The Gaussian kernel size is responsible for the noise-detail trade-off. 



Step 2 — Edge Detection (Module 3) 



The canny edge detection calculates the pixels whose gradient change is above a certain threshold. Then, with the help of non-maximum suppression, Canny emphasizes the edges and minimizes them to one pixel, and finally, with the help of hysteresis with two thresholds, the final connected edge chain is selected. 



Step 3 — Contour Detection (Module 3) 



Contours are searched from the edge map. The contours are then approximated as polygons using the Douglas-Peucker algorithm. Finally, the Douglas-Peucker approximation polygons are inspected, and the largest quadrilateral encompassing most of the image is identified as the document boundary. 



Step 4 — Perspective Transformation (Modules 1 \& 2) 



Once that four corners are identified, a 3×3 homography matrix that maps these corner to a rectangle is computed. The projective transformation enables the perspective-corrected ‘scanning’ of a digital image of a document that is taken at an angle. 



Step 5 — Adaptive Thresholding (Module 1) 



After obtaining the rectified image, it is finally converted to a clean binary (black-on-white) scan. CLAHE (Contrast Limited Adaptive Histogram Equalization) first improved the contrast locally, and then the adaptive Gaussian thresholding binarized the image with locally-computed thresholds, making the output robust to shadows and uneven illumination. 

\---

### Phase 2: Feature Extraction (Module 3)

### 

**ORB Features** 



**Detects corner-like keypoints using FAST, and the keypoints are described using rotation-invariant BRIEF descriptors. The output gives the location of keypoints with their scale and orientation.** 



**SIFT Features** 



**Detects scale-invariant keypoints through Difference of Gaussians in scale space and corresponding descriptors consisting of 128-dimensional gradient histograms. These are scale, rotation, and partially illumination invariant.** 



**HOG Descriptor** 



**Computes the gradient orientation grid histograms, followed by normalization over overlapping blocks. Each corresponding feature vector thus represents the overall shape layout of the document.** 



**Harris Corners** 



**Its corner response function is computed based on eigenvalues of the local structure tensor (gradient auto-correlation matrix). It is only in locations where both eigenvalues are large that corners are detected.** 

**---**

### Phase 3: Segmentation (Modules 3 \& 4)



**K-Means Clustering** 



**It partitions pixels into k color clusters by iteratively assigning pixels to the nearest cluster centroids and recomputing the centroids. This typically distinguishes the text of a document, its background, and graphical elements.** 



**Watershed Segmentation** 



**Treats the image as a topographic surface and ‘floods’ from foreground markers, creating boundaries at points where different flood basins collide. This separates areas which are touching or intersecting.** 



**Text Region Detection** 



**It uses adaptive thresholding, morphological closing (which joins isolated letters into large text blocks) and connected component analysis to identify text areas and enclose them in rectangular regions.** 

\---

## Course Concepts Applied

|Module|Concept|Where Applied|
|-|-|-|
|Module 1|Gaussian Filtering, Convolution|Preprocessing stage|
|Module 1|Histogram Processing (CLAHE)|Contrast enhancement|
|Module 1|Image Enhancement, Restoration|Adaptive thresholding|
|Module 1|Projective Transformation|Perspective warp|
|Module 1|Morphological Operations|Text block merging|
|Module 2|Homography|getPerspectiveTransform|
|Module 3|Canny Edge Detection|Edge map computation|
|Module 3|Harris Corner Detection|Feature extraction|
|Module 3|SIFT, DOG, Scale-Space|Feature extraction|
|Module 3|HOG|Shape descriptor|
|Module 3|Region Growing|Segmentation|
|Module 4|K-Means Clustering|Color segmentation|
|Module 4|Watershed Segmentation|Region segmentation|
|Module 4|Object Detection|Text region detection|

\---

## Output Examples

After running `python main.py demo`, the `output/` directory will contain:

```
output/
├── sample\_input.jpg           # Generated synthetic document photo
├── scan/
│   ├── 01\_detected\_contour.jpg   # Original with green boundary overlay
│   ├── 02\_edges.jpg              # Canny edge map
│   ├── 03\_preprocessed.jpg       # Grayscale + Gaussian blur
│   ├── 04\_warped\_color.jpg       # Perspective-corrected color image
│   ├── 05\_final\_scan.jpg         # Clean binary scan output
│   ├── scanned\_color.jpg         # Final color scan
│   └── scanned\_binary.jpg        # Final binary scan
└── analysis/
    ├── features\_orb.jpg          # ORB keypoints visualization
    ├── features\_sift.jpg         # SIFT keypoints visualization
    ├── features\_harris.jpg       # Harris corners visualization
    ├── seg\_kmeans.jpg            # K-Means color segmentation
    ├── seg\_watershed.jpg         # Watershed boundaries
    └── seg\_text\_regions.jpg      # Detected text regions with bounding boxes
```

\---

## Project Structure

```
smart-doc-scanner/
├── main.py                    # CLI entry point with argparse commands
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── \_\_init\_\_.py
│   ├── scanner.py             # Core scanning pipeline (Modules 1, 2)
│   ├── features.py            # Feature extraction (Module 3)
│   ├── segmentation.py        # Segmentation algorithms (Modules 3, 4)
│   └── visualization.py       # Output generation and visualization
├── samples/                   # Place your test images here
└── output/                    # Generated outputs (created at runtime)
```

\---

## Troubleshooting

**"ModuleNotFoundError: No module named 'cv2'"** — Run `pip install opencv-contrib-python` to install OpenCV with the contrib modules (needed for SIFT).

**"No document found" warning** — The contour detector could not find a clear quadrilateral boundary. This can happen if the document blends into the background or has rounded corners. The system falls back to processing the full image.

**Low keypoint count** — Very small or low-contrast images may yield few keypoints. Try providing a higher-resolution input image.

\---

## License

This project is developed as a course project for CSE3010 — Computer Vision. It is intended for educational purposes.


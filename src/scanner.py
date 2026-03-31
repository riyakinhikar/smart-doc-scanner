"""
Document Scanner Module
-----------------------
Implements the core computer vision pipeline for detecting and extracting
documents from images. This module covers:
  - Image preprocessing (Gaussian blur, morphological operations)
  - Edge detection (Canny)
  - Contour detection and polygon approximation
  - Perspective transformation (homography)

These map directly to Course Modules 1, 2, and 3.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


def preprocess_image(image: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """
    Convert to grayscale, apply Gaussian blur, and perform adaptive thresholding.
    
    Gaussian blur reduces high-frequency noise which would otherwise produce
    spurious edges in the Canny detector. The kernel size controls the
    degree of smoothing — larger values suppress more noise but may lose
    fine detail.
    
    Args:
        image: BGR input image
        blur_ksize: Gaussian kernel size (must be odd)
    
    Returns:
        Preprocessed grayscale image ready for edge detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    return blurred


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve
    contrast in images with uneven lighting. This is especially useful for
    documents photographed under poor lighting conditions.
    
    CLAHE divides the image into small tiles and applies histogram equalization
    to each tile independently, then stitches the results together with
    bilinear interpolation to avoid boundary artifacts.
    
    Covers: Module 1 - Histogram Processing
    
    Args:
        image: Grayscale input image
    
    Returns:
        Contrast-enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


def detect_edges(image: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection to find strong gradients in the image.
    
    The Canny detector works in multiple stages:
      1. Gradient computation using Sobel filters
      2. Non-maximum suppression to thin edges
      3. Hysteresis thresholding with low and high thresholds
    
    Pixels above high_thresh are definite edges; those between low and high
    are kept only if connected to a strong edge pixel.
    
    Covers: Module 3 - Edge Detection (Canny)
    
    Args:
        image: Preprocessed grayscale image
        low_thresh: Lower hysteresis threshold
        high_thresh: Upper hysteresis threshold
    
    Returns:
        Binary edge map
    """
    edges = cv2.Canny(image, low_thresh, high_thresh)
    
    # Dilate edges to close small gaps in the contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


def find_document_contour(edges: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Find the largest quadrilateral contour in the edge map.
    
    The algorithm:
      1. Find all contours in the edge map
      2. Sort by area (descending) to prioritize document-sized regions
      3. Approximate each contour with a polygon (Douglas-Peucker algorithm)
      4. Accept the first quadrilateral (4-vertex polygon) found
    
    This approach is robust because a document in an image will typically
    be the largest rectangular region present.
    
    Covers: Module 3 - Feature Extraction, Module 4 - Object Detection
    
    Args:
        edges: Binary edge map
        image_shape: (height, width) of the original image
    
    Returns:
        Numpy array of 4 corner points, or None if no quadrilateral found
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = image_shape[0] * image_shape[1]
    
    for contour in contours[:10]:  # Only examine the 10 largest contours
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # We want a quadrilateral that covers at least 10% of the image
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 0.1 * img_area:
            return approx.reshape(4, 2)
    
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points in a consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    
    This ordering is essential for the perspective transformation to
    correctly map source corners to destination corners.
    
    The strategy:
      - Top-left has the smallest sum of (x + y)
      - Bottom-right has the largest sum of (x + y)
      - Top-right has the smallest difference of (y - x)
      - Bottom-left has the largest difference of (y - x)
    
    Args:
        pts: Array of 4 points in arbitrary order
    
    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    
    return rect


def perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective (projective) transformation to extract and
    rectify the document region from the image.
    
    This computes a 3x3 homography matrix that maps the four detected
    corner points to a rectangular output. The homography is computed
    using OpenCV's getPerspectiveTransform, which solves the system of
    equations relating source and destination point correspondences.
    
    Covers: Module 1 - Projective Transformation
             Module 2 - Homography
    
    Args:
        image: Original BGR image
        pts: Four corner points of the document
    
    Returns:
        Warped (top-down view) image of the document
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image as the maximum distance
    # between the top pair and bottom pair of corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))
    
    # Compute the height similarly
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))
    
    # Destination points for the top-down view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix (homography)
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Warp the image
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped


def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Convert the warped document image to a clean black-and-white scan
    using adaptive thresholding.
    
    Adaptive thresholding computes a separate threshold for each pixel
    based on its local neighborhood, making it robust to shadows and
    uneven illumination across the document.
    
    Covers: Module 1 - Image Enhancement
    
    Args:
        image: Warped BGR document image
    
    Returns:
        Binary (black and white) document image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement before thresholding
    enhanced = enhance_image(gray)
    
    # Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=7
    )
    
    return binary


def scan_document(image: np.ndarray, interactive: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Complete document scanning pipeline.
    
    Orchestrates the full pipeline:
      1. Preprocess (blur + grayscale)
      2. Edge detection (Canny)
      3. Contour finding (quadrilateral detection)
      4. Perspective transform (homography-based rectification)
      5. Adaptive thresholding (clean scan output)
    
    Args:
        image: BGR input image
        interactive: If True, save intermediate results for visualization
    
    Returns:
        Tuple of (warped_color, warped_binary, metadata_dict)
        metadata_dict contains intermediate results and status info
    """
    metadata = {"status": "success", "steps": {}}
    
    h, w = image.shape[:2]
    
    # Step 1: Preprocess
    preprocessed = preprocess_image(image)
    metadata["steps"]["preprocessed"] = preprocessed
    
    # Step 2: Edge detection
    edges = detect_edges(preprocessed)
    metadata["steps"]["edges"] = edges
    
    # Step 3: Find document contour
    contour = find_document_contour(edges, (h, w))
    
    if contour is None:
        metadata["status"] = "no_document_found"
        metadata["message"] = (
            "Could not detect a quadrilateral document boundary. "
            "Falling back to full-image processing."
        )
        # Fall back: use the entire image
        contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    
    metadata["steps"]["contour"] = contour
    
    # Step 4: Perspective transform
    warped_color = perspective_transform(image, contour)
    metadata["steps"]["warped_color"] = warped_color
    
    # Step 5: Adaptive threshold for clean scan
    warped_binary = apply_adaptive_threshold(warped_color)
    metadata["steps"]["warped_binary"] = warped_binary
    
    return warped_color, warped_binary, metadata

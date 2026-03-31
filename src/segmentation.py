"""
Segmentation Module
--------------------
Implements image segmentation techniques for separating document
regions (text blocks, images, backgrounds) from one another.

Covers: Module 3 - Image Segmentation (Region Growing, Graph-Cut, Mean-Shift)
        Module 4 - Clustering (K-Means, K-Medoids)
"""

import cv2
import numpy as np
from typing import Tuple, List


def kmeans_color_segmentation(image: np.ndarray, k: int = 4, 
                                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an image into k color clusters using K-Means clustering.
    
    K-Means partitions pixel color values into k clusters by iteratively:
      1. Assigning each pixel to its nearest cluster centroid
      2. Recomputing centroids as the mean of all assigned pixels
      3. Repeating until convergence or max iterations
    
    For document analysis, this separates text (dark), background (light),
    and any colored elements into distinct regions.
    
    Covers: Module 4 - K-Means Clustering
    
    Args:
        image: BGR input image
        k: Number of clusters
        max_iter: Maximum iterations for K-Means convergence
    
    Returns:
        Tuple of (segmented_image, labels_map)
    """
    # Reshape image to a 2D array of pixels (each row is a BGR triplet)
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)
    
    # Run K-Means
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    # Convert centers back to uint8
    centers = np.uint8(centers)
    
    # Map each pixel to its cluster center color
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(image.shape)
    
    # Reshape labels to image dimensions
    labels_map = labels.reshape(image.shape[:2])
    
    return segmented_image, labels_map


def watershed_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an image using the Watershed algorithm.
    
    Watershed treats the grayscale image as a topographic surface where
    pixel intensity represents elevation. The algorithm:
      1. Identify sure foreground (via distance transform + thresholding)
      2. Identify sure background (via morphological dilation)
      3. Mark unknown regions between foreground and background
      4. Apply watershed: "flood" from markers, building dams at boundaries
    
    This is particularly effective for separating touching objects such as
    overlapping text characters or adjacent document regions.
    
    Covers: Module 4 - Watershed Segmentation
    
    Args:
        image: BGR input image
    
    Returns:
        Tuple of (marked_image, markers)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding to separate foreground/background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background (dilated region)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground (distance transform + threshold)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background becomes 1, not 0
    markers[unknown == 255] = 0  # Mark unknown regions as 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create visualization: mark boundaries in red
    marked_image = image.copy()
    marked_image[markers == -1] = [0, 0, 255]  # Watershed boundaries in red
    
    return marked_image, markers


def region_growing(image: np.ndarray, seed: Tuple[int, int], 
                   threshold: int = 15) -> np.ndarray:
    """
    Segment a region starting from a seed point using region growing.
    
    Region growing is a bottom-up segmentation approach:
      1. Start from a seed pixel
      2. Examine neighboring pixels
      3. If a neighbor's intensity is within 'threshold' of the region mean,
         add it to the region
      4. Continue until no more pixels can be added
    
    This method is effective when the region of interest has relatively
    uniform intensity (like a text block against a white background).
    
    Covers: Module 3 - Region Growing
    
    Args:
        image: Grayscale input image
        seed: (row, col) coordinates of the seed point
        threshold: Maximum intensity difference for inclusion
    
    Returns:
        Binary mask of the grown region
    """
    h, w = image.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Initialize with seed
    stack = [seed]
    region_sum = float(image[seed[0], seed[1]])
    region_count = 1
    
    visited[seed[0], seed[1]] = True
    mask[seed[0], seed[1]] = 255
    
    # 4-connected neighborhood
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        row, col = stack.pop()
        region_mean = region_sum / region_count
        
        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc
            
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                visited[nr][nc] = True
                pixel_val = float(image[nr, nc])
                
                if abs(pixel_val - region_mean) < threshold:
                    mask[nr, nc] = 255
                    stack.append((nr, nc))
                    region_sum += pixel_val
                    region_count += 1
    
    return mask


def text_region_detection(image: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Detect text regions in a document image using morphological operations
    and connected component analysis.
    
    The approach:
      1. Binarize the image with adaptive thresholding
      2. Apply morphological closing to merge nearby text characters
         into continuous blocks
      3. Find connected components to identify text regions
      4. Filter by aspect ratio and area to keep only valid text blocks
    
    Covers: Module 1 - Morphological Operations
            Module 3 - Object Detection
    
    Args:
        image: BGR input image
    
    Returns:
        Tuple of (annotated_image, list_of_bounding_boxes)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 7
    )
    
    # Morphological closing to merge characters into text blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of text blocks
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = image.copy()
    bboxes = []
    
    img_area = image.shape[0] * image.shape[1]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter: reasonable size and aspect ratio for text
        if area > 0.001 * img_area and w > h * 0.5:
            bboxes.append((x, y, w, h))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return annotated, bboxes


def segment_document_regions(image: np.ndarray) -> dict:
    """
    Run a comprehensive segmentation analysis on the document image.
    
    Combines multiple segmentation approaches to provide a complete
    analysis of the document structure.
    
    Args:
        image: BGR input image
    
    Returns:
        Dictionary with segmentation results from each method
    """
    results = {}
    
    # K-Means color segmentation
    kmeans_img, kmeans_labels = kmeans_color_segmentation(image, k=3)
    results["kmeans"] = {"image": kmeans_img, "labels": kmeans_labels}
    
    # Watershed segmentation
    watershed_img, watershed_markers = watershed_segmentation(image)
    results["watershed"] = {"image": watershed_img, "markers": watershed_markers}
    
    # Text region detection
    text_img, text_bboxes = text_region_detection(image)
    results["text_regions"] = {"image": text_img, "bboxes": text_bboxes}
    
    return results

"""
Visualization Module
---------------------
Generates visual outputs for each stage of the document scanning pipeline.
Saves intermediate results so users can understand and debug the process.
"""

import cv2
import numpy as np
import os
from typing import Optional


def save_pipeline_visualization(image: np.ndarray, metadata: dict,
                                 output_dir: str) -> list:
    """
    Save visual outputs for each step of the scanning pipeline.
    
    Creates annotated images showing:
      1. Original image with detected contour overlay
      2. Edge detection output
      3. Warped (perspective-corrected) color image
      4. Final binary (scanned) output
    
    Args:
        image: Original BGR input image
        metadata: Pipeline metadata from scan_document()
        output_dir: Directory to save visualization images
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 1. Original with contour overlay
    if "contour" in metadata["steps"]:
        contour_vis = image.copy()
        pts = metadata["steps"]["contour"].astype(np.int32)
        cv2.polylines(contour_vis, [pts], True, (0, 255, 0), 3)
        
        # Draw corner circles
        for pt in pts:
            cv2.circle(contour_vis, tuple(pt), 10, (0, 0, 255), -1)
        
        path = os.path.join(output_dir, "01_detected_contour.jpg")
        cv2.imwrite(path, contour_vis)
        saved_files.append(path)
    
    # 2. Edge detection output
    if "edges" in metadata["steps"]:
        path = os.path.join(output_dir, "02_edges.jpg")
        cv2.imwrite(path, metadata["steps"]["edges"])
        saved_files.append(path)
    
    # 3. Preprocessed (grayscale + blur)
    if "preprocessed" in metadata["steps"]:
        path = os.path.join(output_dir, "03_preprocessed.jpg")
        cv2.imwrite(path, metadata["steps"]["preprocessed"])
        saved_files.append(path)
    
    # 4. Warped color
    if "warped_color" in metadata["steps"]:
        path = os.path.join(output_dir, "04_warped_color.jpg")
        cv2.imwrite(path, metadata["steps"]["warped_color"])
        saved_files.append(path)
    
    # 5. Final binary scan
    if "warped_binary" in metadata["steps"]:
        path = os.path.join(output_dir, "05_final_scan.jpg")
        cv2.imwrite(path, metadata["steps"]["warped_binary"])
        saved_files.append(path)
    
    return saved_files


def save_feature_visualization(image: np.ndarray, features: dict,
                                output_dir: str) -> list:
    """
    Save visual outputs for feature extraction results.
    
    Creates annotated images showing:
      - ORB keypoints
      - SIFT keypoints
      - Harris corners
    
    Args:
        image: BGR input image
        features: Feature extraction results from extract_document_features()
        output_dir: Directory to save images
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # ORB keypoints visualization
    if "orb" in features and features["orb"]["keypoints"]:
        orb_vis = cv2.drawKeypoints(
            image, features["orb"]["keypoints"], None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        path = os.path.join(output_dir, "features_orb.jpg")
        cv2.imwrite(path, orb_vis)
        saved_files.append(path)
    
    # SIFT keypoints visualization
    if "sift" in features and features["sift"]["keypoints"]:
        sift_vis = cv2.drawKeypoints(
            image, features["sift"]["keypoints"], None,
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        path = os.path.join(output_dir, "features_sift.jpg")
        cv2.imwrite(path, sift_vis)
        saved_files.append(path)
    
    # Harris corners visualization
    if "harris" in features:
        harris_vis = image.copy()
        harris_vis[features["harris"]["corner_mask"] > 0] = [0, 0, 255]
        path = os.path.join(output_dir, "features_harris.jpg")
        cv2.imwrite(path, harris_vis)
        saved_files.append(path)
    
    return saved_files


def save_segmentation_visualization(seg_results: dict, output_dir: str) -> list:
    """
    Save visual outputs for segmentation results.
    
    Args:
        seg_results: Results from segment_document_regions()
        output_dir: Directory to save images
    
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # K-Means segmentation
    if "kmeans" in seg_results:
        path = os.path.join(output_dir, "seg_kmeans.jpg")
        cv2.imwrite(path, seg_results["kmeans"]["image"])
        saved_files.append(path)
    
    # Watershed segmentation
    if "watershed" in seg_results:
        path = os.path.join(output_dir, "seg_watershed.jpg")
        cv2.imwrite(path, seg_results["watershed"]["image"])
        saved_files.append(path)
    
    # Text region detection
    if "text_regions" in seg_results:
        path = os.path.join(output_dir, "seg_text_regions.jpg")
        cv2.imwrite(path, seg_results["text_regions"]["image"])
        saved_files.append(path)
    
    return saved_files


def create_summary_collage(images: dict, output_path: str, 
                            cols: int = 3, cell_size: int = 400):
    """
    Create a single collage image summarizing all pipeline outputs.
    
    Args:
        images: Dictionary mapping label -> image
        output_path: Path to save the collage
        cols: Number of columns in the grid
        cell_size: Size of each cell in pixels
    """
    n = len(images)
    if n == 0:
        return
    
    rows = (n + cols - 1) // cols
    collage = np.ones((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) * 255
    
    for idx, (label, img) in enumerate(images.items()):
        row = idx // cols
        col = idx % cols
        
        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize to fit cell
        h, w = img.shape[:2]
        scale = min((cell_size - 40) / w, (cell_size - 60) / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Place in cell with centering
        y_start = row * cell_size + 30
        x_start = col * cell_size + (cell_size - new_w) // 2
        
        y_end = min(y_start + new_h, collage.shape[0])
        x_end = min(x_start + new_w, collage.shape[1])
        actual_h = y_end - y_start
        actual_w = x_end - x_start
        
        collage[y_start:y_end, x_start:x_end] = resized[:actual_h, :actual_w]
        
        # Add label text
        text_x = col * cell_size + 10
        text_y = row * cell_size + 20
        cv2.putText(collage, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, collage)

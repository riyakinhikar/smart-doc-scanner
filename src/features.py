"""
Feature Extraction Module
--------------------------
Implements feature detection and description algorithms used for
document analysis, matching, and classification.

Covers: Module 3 - Feature Extraction (SIFT, SURF, HOG, Corners)
        Module 4 - Pattern Analysis
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def detect_orb_features(image: np.ndarray, max_features: int = 500) -> Tuple[list, np.ndarray]:
    """
    Detect ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors.
    
    ORB is a fast, rotation-invariant feature detector that combines
    the FAST keypoint detector with the BRIEF descriptor. It is used
    here as an alternative to SIFT/SURF since it is free from patent
    restrictions and runs efficiently.
    
    The FAST detector identifies corner-like keypoints by examining a
    circle of 16 pixels around each candidate point. BRIEF then
    computes a binary descriptor from intensity comparisons in the
    keypoint neighborhood.
    
    Covers: Module 3 - Corners (Harris-like concepts), Feature Descriptors
    
    Args:
        image: Grayscale input image
        max_features: Maximum number of features to detect
    
    Returns:
        Tuple of (keypoints, descriptors)
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def detect_sift_features(image: np.ndarray, max_features: int = 500) -> Tuple[list, np.ndarray]:
    """
    Detect SIFT (Scale-Invariant Feature Transform) keypoints and descriptors.
    
    SIFT operates through a multi-stage pipeline:
      1. Scale-space extrema detection using Difference of Gaussians (DoG)
      2. Keypoint localization with sub-pixel accuracy
      3. Orientation assignment based on local gradient directions
      4. Descriptor computation using gradient orientation histograms
    
    Each keypoint descriptor is a 128-dimensional vector capturing the
    distribution of gradient orientations in its neighborhood. This
    representation is invariant to scale, rotation, and partially
    invariant to illumination changes.
    
    Covers: Module 3 - SIFT, Scale-Space Analysis, DOG
    
    Args:
        image: Grayscale input image
        max_features: Maximum number of features to retain
    
    Returns:
        Tuple of (keypoints, descriptors)
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def compute_hog_descriptor(image: np.ndarray, 
                            win_size: Tuple[int, int] = (128, 128),
                            cell_size: Tuple[int, int] = (16, 16),
                            block_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) descriptor for an image.
    
    HOG captures the distribution of gradient directions in localized
    portions of the image. The computation proceeds as follows:
      1. Compute gradients using Sobel filters in x and y directions
      2. Divide the image into cells of cell_size pixels
      3. For each cell, build a histogram of gradient orientations (9 bins)
      4. Normalize histograms over overlapping blocks for illumination robustness
      5. Concatenate all block-normalized histograms into one descriptor
    
    HOG descriptors are widely used for object detection (e.g., pedestrian
    detection) because they encode shape information effectively.
    
    Covers: Module 3 - HOG
    
    Args:
        image: Grayscale input image
        win_size: Detection window size (image will be resized to this)
        cell_size: Size of each HOG cell
        block_size: Size of each normalization block
    
    Returns:
        HOG descriptor as a 1D feature vector
    """
    resized = cv2.resize(image, win_size)
    
    block_stride = (cell_size[0], cell_size[1])
    nbins = 9
    
    hog = cv2.HOGDescriptor(
        win_size, block_size, block_stride, cell_size, nbins
    )
    
    descriptor = hog.compute(resized)
    return descriptor.flatten()


def match_features(desc1: np.ndarray, desc2: np.ndarray, 
                   method: str = "bf", ratio_thresh: float = 0.75) -> List:
    """
    Match feature descriptors between two images using brute-force
    matching with Lowe's ratio test.
    
    Lowe's ratio test filters out ambiguous matches by comparing the
    distance to the best match vs the second-best match. If the ratio
    is below the threshold, the match is considered reliable. This
    dramatically reduces false positives.
    
    For binary descriptors (ORB), Hamming distance is used.
    For float descriptors (SIFT), L2 distance is used.
    
    Args:
        desc1: Descriptors from image 1
        desc2: Descriptors from image 2
        method: "bf" for brute-force matching
        ratio_thresh: Lowe's ratio test threshold
    
    Returns:
        List of good matches that pass the ratio test
    """
    if desc1 is None or desc2 is None:
        return []
    
    # Determine norm type based on descriptor type
    if desc1.dtype == np.uint8:
        norm_type = cv2.NORM_HAMMING  # For binary descriptors like ORB
    else:
        norm_type = cv2.NORM_L2  # For float descriptors like SIFT
    
    bf = cv2.BFMatcher(norm_type)
    
    try:
        matches = bf.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    return good_matches


def detect_harris_corners(image: np.ndarray, block_size: int = 2, 
                          ksize: int = 3, k: float = 0.04,
                          threshold_ratio: float = 0.01) -> np.ndarray:
    """
    Detect corners using the Harris corner detector.
    
    The Harris detector computes for each pixel a corner response
    function based on the eigenvalues of the structure tensor (the
    auto-correlation matrix of gradients). For a point to be a corner,
    both eigenvalues must be large, meaning gradients change significantly
    in multiple directions.
    
    The response function R = det(M) - k * trace(M)^2 avoids explicit
    eigenvalue computation. Points where R exceeds a threshold are
    classified as corners.
    
    Covers: Module 3 - Harris Corner Detection
    
    Args:
        image: Grayscale input image
        block_size: Neighborhood size for the structure tensor
        ksize: Sobel kernel size for gradient computation
        k: Harris detector free parameter (typically 0.04-0.06)
        threshold_ratio: Fraction of max response to use as threshold
    
    Returns:
        Binary mask where corners are marked as 255
    """
    image_float = np.float32(image)
    
    # Compute Harris corner response
    harris_response = cv2.cornerHarris(image_float, block_size, ksize, k)
    
    # Dilate to mark the corners more visibly
    harris_response = cv2.dilate(harris_response, None)
    
    # Threshold to get corner locations
    threshold = threshold_ratio * harris_response.max()
    corner_mask = np.zeros_like(image)
    corner_mask[harris_response > threshold] = 255
    
    return corner_mask


def extract_document_features(image: np.ndarray) -> dict:
    """
    Extract a comprehensive set of features from a document image.
    
    This function runs multiple feature extraction algorithms and
    returns all results in a dictionary. This is useful for document
    analysis and classification tasks.
    
    Args:
        image: BGR input image
    
    Returns:
        Dictionary with keys: 'orb', 'sift', 'hog', 'harris', 'stats'
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = {}
    
    # ORB features
    orb_kp, orb_desc = detect_orb_features(gray)
    results["orb"] = {"keypoints": orb_kp, "descriptors": orb_desc}
    
    # SIFT features
    sift_kp, sift_desc = detect_sift_features(gray)
    results["sift"] = {"keypoints": sift_kp, "descriptors": sift_desc}
    
    # HOG descriptor
    hog_desc = compute_hog_descriptor(gray)
    results["hog"] = {"descriptor": hog_desc}
    
    # Harris corners
    harris_mask = detect_harris_corners(gray)
    results["harris"] = {"corner_mask": harris_mask}
    
    # Basic statistics
    results["stats"] = {
        "orb_keypoints": len(orb_kp),
        "sift_keypoints": len(sift_kp),
        "hog_dimensions": len(hog_desc),
        "harris_corners": int(np.sum(harris_mask > 0))
    }
    
    return results

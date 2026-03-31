"""
Smart Document Scanner - Main CLI Entry Point
=============================================
A command-line document scanning application that demonstrates core
Computer Vision techniques from CSE3010.

Usage:
    python main.py scan <image_path> [--output <dir>]
    python main.py analyze <image_path> [--output <dir>]
    python main.py demo [--output <dir>]
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scanner import scan_document, preprocess_image, detect_edges
from features import extract_document_features
from segmentation import segment_document_regions
from visualization import (
    save_pipeline_visualization,
    save_feature_visualization,
    save_segmentation_visualization,
    create_summary_collage,
)


def generate_sample_document() -> np.ndarray:
    """
    Generate a synthetic sample document image for demonstration purposes.
    
    This creates a realistic test scenario: a white rectangular "document"
    placed on a colored background with a slight perspective distortion,
    simulating a photo of a document on a desk.
    
    Returns:
        BGR image containing a synthetic document
    """
    # Create a background (desk surface)
    bg = np.ones((800, 1000, 3), dtype=np.uint8)
    bg[:] = (180, 160, 140)  # Warm brownish desk color
    
    # Add some texture to the background
    noise = np.random.randint(0, 20, bg.shape, dtype=np.uint8)
    bg = cv2.add(bg, noise)
    
    # Create a white "document" with text-like content
    doc = np.ones((500, 380, 3), dtype=np.uint8) * 250
    
    # Add a header
    cv2.putText(doc, "QUARTERLY REPORT", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.line(doc, (30, 60), (350, 60), (20, 20, 20), 2)
    
    # Add body text lines
    texts = [
        "Revenue: $2.4M (+12%)",
        "Operating Costs: $1.1M",
        "Net Profit: $1.3M",
        "",
        "Key Highlights:",
        "- Market share grew 5%",
        "- New product launched Q2",
        "- Customer base: 12,400",
        "",
        "Outlook: Positive growth",
        "expected in next quarter",
        "with new partnerships.",
    ]
    y = 100
    for text in texts:
        cv2.putText(doc, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        y += 30
    
    # Add a simple chart (bar chart)
    bars_x = [50, 120, 190, 260]
    bars_h = [80, 110, 95, 130]
    for x, h in zip(bars_x, bars_h):
        cv2.rectangle(doc, (x, 470 - h), (x + 50, 470), (70, 130, 70), -1)
        cv2.rectangle(doc, (x, 470 - h), (x + 50, 470), (40, 90, 40), 1)
    
    cv2.putText(doc, "Q1  Q2  Q3  Q4", (55, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    
    # Define source (document corners) and destination (perspective) points
    src_pts = np.array([[0, 0], [380, 0], [380, 500], [0, 500]], dtype=np.float32)
    
    # Perspective-distorted position on the background
    dst_pts = np.array([
        [200, 80],    # top-left
        [750, 120],   # top-right
        [780, 680],   # bottom-right
        [150, 720],   # bottom-left
    ], dtype=np.float32)
    
    # Compute homography and warp the document onto the background
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc, M, (1000, 800))
    
    # Create a mask for blending
    mask = np.ones((500, 380), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (1000, 800))
    
    # Blend document onto background
    result = bg.copy()
    result[warped_mask > 0] = warped_doc[warped_mask > 0]
    
    # Add slight shadow along document edges
    kernel = np.ones((7, 7), np.uint8)
    shadow_mask = cv2.dilate(warped_mask, kernel, iterations=2) - warped_mask
    result[shadow_mask > 0] = (result[shadow_mask > 0] * 0.7).astype(np.uint8)
    
    return result


def cmd_scan(args):
    """Execute the document scanning pipeline."""
    print("=" * 60)
    print("  Smart Document Scanner - Scan Mode")
    print("=" * 60)
    
    # Load image
    if args.image == "demo":
        print("\n[INFO] Using generated sample document image.")
        image = generate_sample_document()
        sample_path = os.path.join(args.output, "sample_input.jpg")
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(sample_path, image)
        print(f"[INFO] Sample image saved to: {sample_path}")
    else:
        if not os.path.isfile(args.image):
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        print(f"\n[INFO] Loading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Could not read image: {args.image}")
            sys.exit(1)
    
    h, w = image.shape[:2]
    print(f"[INFO] Image dimensions: {w} x {h}")
    
    # Run scanning pipeline
    print("\n--- Running Scanning Pipeline ---")
    start = time.time()
    
    print("[STEP 1/5] Preprocessing (Gaussian blur + grayscale)...")
    print("[STEP 2/5] Edge detection (Canny)...")
    print("[STEP 3/5] Document contour detection...")
    print("[STEP 4/5] Perspective transformation (homography)...")
    print("[STEP 5/5] Adaptive thresholding...")
    
    warped_color, warped_binary, metadata = scan_document(image)
    
    elapsed = time.time() - start
    print(f"\n[INFO] Pipeline completed in {elapsed:.2f}s")
    print(f"[INFO] Status: {metadata['status']}")
    if "message" in metadata:
        print(f"[INFO] {metadata['message']}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    if warped_color is not None:
        color_path = os.path.join(args.output, "scanned_color.jpg")
        cv2.imwrite(color_path, warped_color)
        print(f"\n[OUTPUT] Color scan:  {color_path}")
    
    if warped_binary is not None:
        binary_path = os.path.join(args.output, "scanned_binary.jpg")
        cv2.imwrite(binary_path, warped_binary)
        print(f"[OUTPUT] Binary scan: {binary_path}")
    
    # Save pipeline visualization
    print("\n--- Saving Pipeline Visualizations ---")
    vis_files = save_pipeline_visualization(image, metadata, args.output)
    for f in vis_files:
        print(f"[OUTPUT] {os.path.basename(f)}")
    
    print(f"\n[DONE] All outputs saved to: {args.output}/")


def cmd_analyze(args):
    """Execute feature extraction and segmentation analysis."""
    print("=" * 60)
    print("  Smart Document Scanner - Analyze Mode")
    print("=" * 60)
    
    # Load image
    if args.image == "demo":
        print("\n[INFO] Using generated sample document image.")
        image = generate_sample_document()
    else:
        if not os.path.isfile(args.image):
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        print(f"\n[INFO] Loading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Could not read image: {args.image}")
            sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Feature extraction
    print("\n--- Feature Extraction ---")
    start = time.time()
    features = extract_document_features(image)
    elapsed = time.time() - start
    
    stats = features["stats"]
    print(f"  ORB keypoints detected:    {stats['orb_keypoints']}")
    print(f"  SIFT keypoints detected:   {stats['sift_keypoints']}")
    print(f"  HOG descriptor dimensions: {stats['hog_dimensions']}")
    print(f"  Harris corners detected:   {stats['harris_corners']}")
    print(f"  Time: {elapsed:.2f}s")
    
    # Save feature visualizations
    feat_files = save_feature_visualization(image, features, args.output)
    for f in feat_files:
        print(f"  [OUTPUT] {os.path.basename(f)}")
    
    # Segmentation
    print("\n--- Segmentation Analysis ---")
    start = time.time()
    seg_results = segment_document_regions(image)
    elapsed = time.time() - start
    
    if "text_regions" in seg_results:
        n_boxes = len(seg_results["text_regions"]["bboxes"])
        print(f"  Text regions detected: {n_boxes}")
    
    print(f"  Methods run: K-Means, Watershed, Text Region Detection")
    print(f"  Time: {elapsed:.2f}s")
    
    # Save segmentation visualizations
    seg_files = save_segmentation_visualization(seg_results, args.output)
    for f in seg_files:
        print(f"  [OUTPUT] {os.path.basename(f)}")
    
    # Create summary collage
    print("\n--- Creating Summary Collage ---")
    collage_images = {}
    if "edges" in seg_results.get("_meta", {}):
        collage_images["Edges"] = seg_results["_meta"]["edges"]
    if "kmeans" in seg_results:
        collage_images["K-Means"] = seg_results["kmeans"]["image"]
    if "watershed" in seg_results:
        collage_images["Watershed"] = seg_results["watershed"]["image"]
    if "text_regions" in seg_results:
        collage_images["Text Regions"] = seg_results["text_regions"]["image"]
    
    if collage_images:
        collage_path = os.path.join(args.output, "analysis_summary.jpg")
        create_summary_collage(collage_images, collage_path)
        print(f"  [OUTPUT] analysis_summary.jpg")
    
    print(f"\n[DONE] All outputs saved to: {args.output}/")


def cmd_demo(args):
    """Run both scan and analyze on the generated sample."""
    print("=" * 60)
    print("  Smart Document Scanner - Full Demo")
    print("=" * 60)
    print("\nThis demo generates a synthetic document image and runs")
    print("the complete CV pipeline (scanning + analysis) on it.\n")
    
    # Generate sample
    image = generate_sample_document()
    os.makedirs(args.output, exist_ok=True)
    sample_path = os.path.join(args.output, "sample_input.jpg")
    cv2.imwrite(sample_path, image)
    print(f"[INFO] Sample image saved: {sample_path}")
    
    # Run scan
    print("\n" + "-" * 40)
    print("  PHASE 1: Document Scanning")
    print("-" * 40)
    
    warped_color, warped_binary, metadata = scan_document(image)
    print(f"  Status: {metadata['status']}")
    
    scan_dir = os.path.join(args.output, "scan")
    os.makedirs(scan_dir, exist_ok=True)
    
    if warped_color is not None:
        cv2.imwrite(os.path.join(scan_dir, "scanned_color.jpg"), warped_color)
    if warped_binary is not None:
        cv2.imwrite(os.path.join(scan_dir, "scanned_binary.jpg"), warped_binary)
    
    vis_files = save_pipeline_visualization(image, metadata, scan_dir)
    print(f"  Saved {len(vis_files)} pipeline images to {scan_dir}/")
    
    # Run analysis
    print("\n" + "-" * 40)
    print("  PHASE 2: Feature & Segmentation Analysis")
    print("-" * 40)
    
    analysis_dir = os.path.join(args.output, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    features = extract_document_features(image)
    stats = features["stats"]
    print(f"  ORB keypoints:    {stats['orb_keypoints']}")
    print(f"  SIFT keypoints:   {stats['sift_keypoints']}")
    print(f"  HOG dimensions:   {stats['hog_dimensions']}")
    print(f"  Harris corners:   {stats['harris_corners']}")
    
    feat_files = save_feature_visualization(image, features, analysis_dir)
    
    seg_results = segment_document_regions(image)
    seg_files = save_segmentation_visualization(seg_results, analysis_dir)
    
    total_outputs = len(vis_files) + len(feat_files) + len(seg_files) + 3
    print(f"  Saved {len(feat_files) + len(seg_files)} analysis images to {analysis_dir}/")
    
    print(f"\n{'=' * 60}")
    print(f"  Demo complete! {total_outputs} files generated.")
    print(f"  Output directory: {args.output}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Document Scanner - A Computer Vision CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                           Run full demo with synthetic image
  python main.py scan photo.jpg                 Scan a document from a photo
  python main.py scan photo.jpg -o results/     Scan and save to results/
  python main.py analyze document.jpg           Extract features & segment regions
  python main.py scan demo                      Scan the generated sample image
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a document from an image")
    scan_parser.add_argument("image", help="Path to input image (or 'demo' for sample)")
    scan_parser.add_argument("-o", "--output", default="output",
                             help="Output directory (default: output/)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Extract features and segment regions")
    analyze_parser.add_argument("image", help="Path to input image (or 'demo' for sample)")
    analyze_parser.add_argument("-o", "--output", default="output",
                                help="Output directory (default: output/)")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run full demo with synthetic image")
    demo_parser.add_argument("-o", "--output", default="output",
                             help="Output directory (default: output/)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()

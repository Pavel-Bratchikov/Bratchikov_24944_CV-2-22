import argparse
import os
from typing import Tuple
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people


def nms(boxes):
    """
    Perform Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    This function eliminates redundant bounding boxes that significantly overlap
    with larger ones, keeping only the most confident detections. It uses the
    Intersection over Union (IoU) metric to decide whether to keep or suppress a box.

    Args:
        boxes (list): A list of bounding boxes in the format [x, y, w, h].

    Returns:
        list: A filtered list of bounding boxes after applying NMS.
    """
    if not boxes:
        return []
    
    boxes = np.array(boxes)
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = np.argsort(areas)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= 0.3)[0]
        order = order[inds + 1]

    return boxes[keep].tolist()


def load_image_from_path(path: str) -> Tuple[np.ndarray, str]:
    """
    Load an image from a given path and convert it from BGR to RGB format.

    OpenCV loads images in BGR format by default, but many visualization
    libraries (e.g., matplotlib) use RGB, so conversion is necessary.

    Args:
        path (str): Path to the image file.

    Returns:
        Tuple[np.ndarray, str]:
            - The image in RGB format (numpy array).
            - The base name of the file (string).

    Raises:
        FileNotFoundError: If the image file cannot be loaded.
    """
    bgr = cv2.imread(path)

    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb, os.path.basename(path)


def improve_gray_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Enhance the contrast of a grayscale image using CLAHE.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local
    contrast and reduces the effect of non-uniform lighting, making it
    easier for face detection algorithms to identify facial features.

    Args:
        gray (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: The contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return clahe.apply(gray)


def main():
    """
    Main function for face detection with optional eyes and profile detection.

    Steps:
        1. Parse command-line arguments.
        2. Load image either from local path or from the LFW dataset.
        3. Convert to grayscale and apply preprocessing (CLAHE + Gaussian blur).
        4. Detect frontal faces using Haar Cascade.
        5. Optionally detect profile faces.
        6. Optionally filter faces by checking for eyes.
        7. Apply Non-Maximum Suppression (NMS) to remove duplicate detections.
        8. Draw bounding boxes around detected faces.
        9. Save the result to the output file.

    Command-line arguments:
        --image, -i (str): Path to a local image file.
        --lfw_index (int): Index of an image from sklearn.datasets.fetch_lfw_people.
        --scale (float): Scale factor for face detection pyramid (default=1.1).
        --neighbors (int): Min neighbors parameter for detection (default=5).
        --save (str): Output file name for saving the processed image (required).
        --check_eyes (flag): If set, keeps only faces with at least one detected eye.
        --with_profiles (flag): If set, detects profile faces in addition to frontal.

    Prints:
        - Number of detected faces.
        - Confirmation message with the path of the saved result.
    """
    p = argparse.ArgumentParser(description="Face detection with optional eyes and profile detection.")
    p.add_argument("--image", "-i", type=str, help="Path to your image.")
    p.add_argument("--lfw_index", type=int, help="Index of image from sklearn.datasets.fetch_lfw_people.")
    p.add_argument("--scale", type=float, default=1.1, help="scaleFactor for detectMultiScale (default=1.1).")
    p.add_argument("--neighbors", type=int, default=5, help="minNeighbors for detectMultiScale (default=5).")
    p.add_argument("--save", type=str, required=True, help="Output file name to save the result (required).")
    p.add_argument("--check_eyes", action="store_true", help="Filter detected faces by presence of eyes.")
    p.add_argument("--with_profiles", action="store_true", help="Detect profile faces in addition to frontal.")
    args = p.parse_args()

    # Load image
    if args.lfw_index is not None:
        lfw = fetch_lfw_people(color=True, resize=0.5)
        rgb = (lfw.images[args.lfw_index] * 255).astype(np.uint8)

    elif args.image is not None:
        rgb, _ = load_image_from_path(args.image)
    else:
        raise ValueError("Either --image or --lfw_index must be provided")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = improve_gray_contrast(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Frontal face detection
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_xml)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale,
        minNeighbors=args.neighbors,
        minSize=(50, 50)
    )

    faces = list(faces)

    # Optional profile face detection
    if args.with_profiles:
        prof_xml = cv2.data.haarcascades + "haarcascade_profileface.xml"
        prof_cascade = cv2.CascadeClassifier(prof_xml)

        prof_faces = prof_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale,
            minNeighbors=args.neighbors,
            minSize=(50, 50)
        )

        faces.extend(prof_faces)

    # Optional eyes check
    if args.check_eyes:
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(eye_xml)
        filtered_faces = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(int(w * 0.15), int(h * 0.15))
            )

            if len(eyes) >= 1:
                filtered_faces.append((x, y, w, h))

        faces = filtered_faces

    # Apply NMS
    faces = nms(faces)

    # Fallback if no face detected in LFW
    if args.lfw_index is not None and len(faces) == 0:
        h, w = rgb.shape[:2]
        faces = [(0, 0, w, h)]
        print("Cascade did not find a face, using entire image as face.")

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Found {len(faces)} face(s).")

    # Save result
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.save, bgr)
    print(f"Saved result to {args.save}")


if __name__ == "__main__":
    main()
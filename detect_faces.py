import argparse, os
from typing import Tuple
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people


def nms(boxes, iou_thr=0.3):
    """
    Perform Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Args:
        boxes (list): List of bounding boxes [x, y, w, h].
        iou_thr (float): IoU threshold for suppression.

    Returns:
        list: Filtered bounding boxes.
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
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return boxes[keep].tolist()


def load_image_from_path(path: str) -> Tuple[np.ndarray, str]:
    bgr = cv2.imread(path)

    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb, os.path.basename(path)


def improve_gray_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    return clahe.apply(gray)


def main():
    p = argparse.ArgumentParser(description="Face detection with optional eyes and profile detection.")
    p.add_argument("--image", "-i", type=str, help="Path to your image.")
    p.add_argument("--lfw_index", type=int, help="Index of image from sklearn.datasets.fetch_lfw_people.")
    p.add_argument("--scale", type=float, default=1.1, help="scaleFactor for detectMultiScale (default=1.1).")
    p.add_argument("--neighbors", type=int, default=5, help="minNeighbors for detectMultiScale (default=5).")
    p.add_argument("--save", type=str, required=True, help="Output file name to save the result (required).")
    p.add_argument("--minsize", type=int, default=50, help="Minimum face size in pixels.")
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
        minSize=(args.minsize, args.minsize)
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
            minSize=(args.minsize, args.minsize)
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
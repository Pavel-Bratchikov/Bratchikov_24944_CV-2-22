# Face Detection Script

This script is a simplified and enhanced adaptation of an existing GitHub project that uses OpenCV Haar cascades for face detection. It supports both local images and the Labeled Faces in the Wild (LFW) dataset.

## Features

- Detects frontal faces and optionally profile faces (`--with_profiles`).
- Optionally filters faces by the presence of eyes (`--check_eyes`).
- Preprocesses images with CLAHE contrast enhancement and Gaussian blur.
- Removes duplicate detections using Non-Maximum Suppression (NMS).
- Works with local images or LFW dataset images (`--lfw_index`).
- Saves detected faces highlighted in green rectangles.

## Key Improvements Over the Original Project

- Added support for LFW dataset.
- Minimum face size set to 50px.
- Robust fallback: if no faces are detected in LFW, the entire image is used.
- Simplified workflow without matplotlib visualization.

## Usage

```bash
# Detect faces in a local image
python face_detect.py --image path/to/image.jpg --save output.png

# Detect faces in an LFW dataset image with eye verification
python face_detect.py --lfw_index 5 --check_eyes --save output.png

# Detect frontal and profile faces
python face_detect.py --image path/to/image.jpg --with_profiles --save output.png

 

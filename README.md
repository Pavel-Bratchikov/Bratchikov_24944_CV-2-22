# CV-2-22
This script is an improved version of the solution to problem CV-1-11 https://github.com/costo4ka/itpw/tree/main/task1. I strongly recommend that you first read that solution for a better understanding.

## Key Improvements Over the Original Project

- Added support for LFW dataset.
- Minimum face size set to 50px.
- Robust fallback: if no faces are detected in LFW, the entire image is used.
- Simplified workflow without matplotlib visualization.

## 1. Installation
 ```bash
git clone https://github.com/Pavel-Bratchikov/Bratchikov_24944_CV-2-22.git
cd Bratchikov_24944_CV-2-22
```

## 2. Create a virtual environment (optional but recommended)
Windows:
```
python -m venv venv
venv\Scripts\activate
```

Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage
The script can be run from the command line with different options:
```bash
python detect_faces.py --image path/to/image.jpg --save output.png
python detect_faces.py --lfw_index 5 --check_eyes --save output.png
python detect_faces.py --image path/to/image.jpg --with_profiles --save output.png
```

Command-line arguments: 
--image or -i: path to the local image to process. Either this or --lfw_index must be provided.  
--lfw_index: index of an image from the LFW dataset (fetch_lfw_people).
--scale: scaleFactor for detectMultiScale. Default is 1.1. Smaller = more accurate detection.
--neighbors: minNeighbors for detectMultiScale. Default is 5. Higher = fewer false positives.
--minsize: minimum face size in pixels. Default is 50.
--check_eyes: optional flag. If present, filters detected faces to keep only those with at least one detected eye.
--with_profiles: optional flag. If present, detects profile faces in addition to frontal faces.
--save: output file name for the processed image. This argument is required.

Exmple for Windows:
```bash
python face_detect.py --image group.jpg --save result.png
```

Example for Linux/macOS:
```bash
python3 face_detect.py --image group.jpg --save result.png
```

## Notes
Ensure the virtual environment is activated before running the script.
The script saves the processed image with green rectangles around detected faces.
If no faces are found in an LFW image, the script uses the entire image as a fallback.

# Auto Labeling Program

## 1. What is Labeling?
Labeling is the process of creating an answer key for given data, and this answer key is referred to as a **label**. In **supervised learning**, a form of deep learning, labeled data is required. If the model is trained with inaccurate labels, its performance will degrade, making **accurate labeling** crucial.

## 2. Motivation
In ROS-based autonomous driving systems, data labeling for sign recognition is an essential task. However, manual labeling can be time-consuming. To efficiently automate this process, the auto-labeling program was developed. This program aims to significantly reduce development time by automating the recognition and labeling of signs.

## 3. Program Description

### 3.1 Labeling Targets
The auto-labeling program targets the following types of signs:
- Intersection
- Construction Zone
- Level Crossing
- Parking
- Tunnel

### 3.2 Purpose of Auto Labeling
The program is designed to quickly and accurately detect various sign objects and automatically generate data for machine learning model training.

## 4. Program Structure

### 4.1 Initial Setup
- Capture video stream from a camera.
- Set the width and height of the frame.
- Store class information for labeling in a `classes.txt` file.

### 4.2 Sign Detection Process
1. **Intersection Sign Detection**
   - Convert the image to grayscale and use Hough Transform to detect circles. Based on the position and size of the circles, create a label file.
   
2. **Construction & Tunnel Sign Detection**
   - Convert the image to HSV color space, combine yellow and black areas to emphasize boundaries, and detect triangles to create the label file.
   
3. **Stop Sign Detection**
   - Extract the red area from the image, detect circles, and create a label file based on the position and size.
   
4. **Parking Sign Detection**
   - Extract the blue area, detect rectangles, and create a label file based on the position and size.
   
5. **Program Termination**
   - Close the camera and all OpenCV windows.

## 5. Source Code Explanation

### 5.1 Intersection Sign Detection
- **Frame Reading**: Use `capture.read()` to read frames from the camera.
- **Grayscale Conversion**: Convert the image to grayscale using `cv2.cvtColor()`.
- **Hough Transform**: Detect circles using `cv2.HoughCircles()` and store the coordinates and size.

### 5.2 Construction & Tunnel Sign Detection
- **HSV Conversion and Color Extraction**: Convert the image to HSV color space, then extract yellow and black areas to highlight the boundaries.
- **Triangle Detection and Storage**: Extract contours, detect triangles, and store their position and size.

## 6. Additional Activities

### 6.1 Model Performance Improvement
Brightness adjustment and rotation features were added to diversify the training data, resulting in improved model performance.

### 6.2 Future Improvements
In the future, the program will be enhanced with an improved algorithm capable of detecting multiple signs in various environments simultaneously.



If you have any questions, feel free to contact me at Doyoung Kim (felixkim0719@gmail.com).

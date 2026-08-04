# Task2_Color.Recognition_SmartMethods.
A real-time color detection system built using OpenCV. The program captures video from the webcam, detects Red, Green, and Blue colors using HSV thresholds, draws bounding boxes around detected objects, and labels each color on screen. The project runs inside an Anaconda virtual environment using Visual Studio.
# 🎨 Real-Time Color Detection using OpenCV

This project implements real-time color detection using the OpenCV library.  
The system identifies Red, Green, and Blue objects from a live webcam feed, draws bounding boxes around them, and displays the detected color name.

---

## 📌 Features
- Real-time webcam capture  
- HSV color filtering  
- Detection of Red, Green, and Blue  
- Contour extraction  
- Bounding box + label display  
- Noise filtering using contour area  

---

## 🧰 Technologies Used
- Python  
- OpenCV  
- NumPy  
- Anaconda (virtual environment)  
- Visual Studio (execution environment)

---

## 🧠 How It Works
1. The webcam feed is captured using cv2.VideoCapture(0).
2. Each frame is converted from BGR → HSV.
3. HSV ranges for Red, Green, and Blue are defined.
4. Binary masks are created for each color.
5. Contours are extracted from each mask.
6. Large contours (area > 500) are considered valid detections.
7. A bounding rectangle and color label are drawn on the frame.

---

## 🖥 Code

```python
import cv2
import numpy as np

# Open the default webcam
cap = cv2.VideoCapture(0)

# Create OpenCV window
cv2.namedWindow("Color Detection", cv2.WINDOW_NORMAL)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Exit the loop if no frame is captured
    if not ret:
        break
        
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for the color Red
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Define the HSV range for the color Green
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Define the HSV range for the color Blue
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create binary masks for each color
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)
               
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Store masks with colors and labels
    masks = [
        (mask_red, (0, 0, 255), "Red"),
        (mask_green, (0, 255, 0), "Green"),
        (mask_blue, (255, 0, 0), "Blue")
    ]
    
    # Process each color mask
    for mask, color, name in masks:
        
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Detect objects
        for cnt in contours:
            
            # Ignore small noise
            if cv2.contourArea(cnt) > 500:
                
                # Get rectangle coordinates
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )
                
                # Display color name
                cv2.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
    # Show camera output
    cv2.imshow("Color Detection", frame)
    
    # Close with q or ESC
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        break

# Release webcam
cap.release()

# Close OpenCV windows safely
cv2.destroyAllWindows()
cv2.waitKey(1)

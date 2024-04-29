import matplotlib.pyplot as plt
import cv2

def detect_faces(image):
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Return the coordinates of the detected faces
    return faces

# Example usage
image_path = 'test3.jpeg'
image = cv2.imread(image_path)
faces = detect_faces(image)
print("Detected faces:")
for (x, y, w, h) in faces:
    print("Face found at (x={}, y={}), width={}, height={}".format(x, y, w, h))
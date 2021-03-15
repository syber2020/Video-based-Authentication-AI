import cv2
import numpy as np
import os
# /Users/Syed/UNT/SPRING2021/5214/vandana/Video-Auth-P2/
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

new_dir ="test"
parent_dir = "/Users/Syed/UNT/SPRING2021/5214/vandana/Video-Auth-P2/images/"
new_path = os.path.join(parent_dir, new_dir)
os.mkdir(new_path)

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        face = cv2.cvtColor(face_extractor(frame), cv2.COLOR_RGBA2RGB )



        # Save file in specified directory with unique name
        file_name = 'train' + str(count) + '.jpg'

        # cv2.imwrite(file_name, face)

        cv2.imwrite(os.path.join(new_path ,file_name), face)
        # Put count on images and display live count
        cv2.putText(face, str(count), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if count == 10 : #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")

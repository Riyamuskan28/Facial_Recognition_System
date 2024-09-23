import cv2
import os
import numpy as np
import face_recognition

# Dictionary to store reference encodings and names
known_faces = {
    "Shraddha kapoor": "C:/Users/dell/Desktop/imagerecog/sr.png",  # Add your reference image path here
    "Alia bhatt": "C:/Users/dell/Desktop/imagerecog/alia.png",
    "MS Dhoni":"C:/Users/dell/Desktop/imagerecog/thala.jpg"
}

def load_known_faces():
    known_encodings = {}
    for name, image_path in known_faces.items():
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings[name] = encoding
    return known_encodings

def face_recognition_on_image(known_encodings, image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding)
        name = "Unknown"

        # If a match is found, get the name
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_encodings.keys())[first_match_index]
            print("Match found:", name)
        else:
            print("No match found")

        top, right, bottom, left = face_location
        
        # Choose color based on whether the person is known or unknown
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for known
        
        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load known faces
known_encodings = load_known_faces()
image_path_to_compare = "C:/Users/dell/Desktop/imagerecog/ms.png"
face_recognition_on_image(known_encodings, image_path_to_compare)

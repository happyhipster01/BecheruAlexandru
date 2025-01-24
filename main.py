import cv2
import mediapipe as mp

# Inițializam MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Incarcm imaginea
image_path = r'C:\Users\Alex\PycharmProjects\PythonProject2\Images\people2.jpg'
image = cv2.imread(image_path)

# Convertim imaginea la RGB (MediaPipe folosește acest format)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectam fețele folosind MediaPipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    results = face_detection.process(rgb_image)

    # Daca exista fețe detectate, le desenam pe imagine
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

# Afișam imaginea cu fețele detectate
cv2.imshow('Face Detection', image)

# Așteptam apasarea unei taste pentru a inchide fereastra
cv2.waitKey(0)
cv2.destroyAllWindows()
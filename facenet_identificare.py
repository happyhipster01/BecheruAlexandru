import os
import numpy as np
import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def load_facenet_model():
    return FaceNet()


def extract_face(image_path, detector, required_size=(160, 160)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        x, y = abs(x), abs(y)
        face = image[y:y + height, x:x + width]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        return np.asarray(image)
    return None


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    embedding = model.embeddings(np.expand_dims(face_pixels, axis=0))
    return embedding[0]


def analyze_faces(image_folder):
    detector = MTCNN()
    model = load_facenet_model()
    embeddings = {}

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        face = extract_face(image_path, detector)
        if face is not None:
            embedding = get_embedding(model, face)
            embeddings[image_name] = embedding

    return embeddings


def find_matching_faces(input_image_path, embeddings, model, threshold=0.5):
    detector = MTCNN()
    face = extract_face(input_image_path, detector)
    if face is None:
        print("No face detected in input image.")
        return []

    input_embedding = get_embedding(model, face)
    matching_faces = []

    for name, emb in embeddings.items():
        similarity = cosine_similarity([input_embedding], [emb])[0][0]
        if similarity > threshold:
            matching_faces.append((name, similarity))

    return sorted(matching_faces, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    image_folder = "C:/Users/alex/PycharmProjects/PythonProject5/Images"
    test_image_path = "C:/Users/alex/PycharmProjects/PythonProject5/Test/test_image.jpeg"

    model = load_facenet_model()
    embeddings = analyze_faces(image_folder)
    matches = find_matching_faces(test_image_path, embeddings, model)

    if matches:
        print("Matching faces:")
        for name, score in matches:
            print(f"{name}: {score:.2f}")
    else:
        print("No matching faces found.")
import os
import numpy as np
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


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


def get_embedding(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
    return np.array(embedding[0]['embedding'])


def analyze_faces(image_folder):
    detector = MTCNN()
    embeddings = {}

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        face = extract_face(image_path, detector)
        if face is not None:
            embedding = get_embedding(image_path)
            embeddings[image_name] = embedding

    return embeddings


def find_matching_faces(input_image_path, embeddings, threshold=0.5):
    input_embedding = get_embedding(input_image_path)
    matching_faces = []

    for name, emb in embeddings.items():
        similarity = cosine_similarity([input_embedding], [emb])[0][0]
        if similarity > threshold:
            matching_faces.append((name, similarity))

    return sorted(matching_faces, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    image_folder = "C:/Users/alex/PycharmProjects/PythonProject6/Images"
    test_image_path = "C:/Users/alex/PycharmProjects/PythonProject6/Test/test_image.jpeg"

    embeddings = analyze_faces(image_folder)
    matches = find_matching_faces(test_image_path, embeddings)

    if matches:
        print("Matching faces:")
        for name, score in matches:
            print(f"{name}: {score:.2f}")
    else:
        print("No matching faces found.")

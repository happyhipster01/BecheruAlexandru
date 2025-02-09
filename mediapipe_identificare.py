import os
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Inițializare MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_model = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def extract_face(image_path, required_size=(160, 160)):
    """Extrage fața din imagine și o redimensionează."""
    image = cv2.imread(image_path)
    if image is None:
        return None  # Imaginea nu a fost încărcată corect

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_model.process(image)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h))
            x, y = max(x, 0), max(y, 0)
            face = image[y:y + height, x:x + width]

            if face.size == 0:
                return None  # Evităm erori în cazul unei decupări incorecte

            image_resized = Image.fromarray(face).resize(required_size)
            return np.asarray(image_resized, dtype=np.float32) / 255.0  # Normalizare între 0 și 1

    return None


def get_embedding(image_path):
    """Obține vectorul de trăsături al feței folosind MediaPipe."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_model.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            return np.array([bbox.xmin, bbox.ymin, bbox.width, bbox.height], dtype=np.float32)  # Vector numeric fix
    return None


def analyze_faces(image_folder):
    """Analizează toate fețele dintr-un folder și returnează embeddings."""
    if not os.path.exists(image_folder):
        print(f"Folderul specificat nu există: {image_folder}")
        return {}

    embeddings = {}

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        embedding = get_embedding(image_path)

        if embedding is not None:
            embeddings[image_name] = embedding

    return embeddings


def match_face(target_image, database_embeddings, threshold=0.5):
    """Compară o imagine de referință cu baza de date și returnează cele mai bune potriviri."""
    target_embedding = get_embedding(target_image)

    if target_embedding is None:
        print("Nu s-a putut extrage fața din imaginea de referință.")
        return []

    similarities = {}

    for name, emb in database_embeddings.items():
        similarity = cosine_similarity([target_embedding], [emb])[0][0]
        if similarity >= threshold:  # Doar imaginile peste un anumit prag sunt afișate
            similarities[name] = similarity

    # Sortare descrescătoare după similaritate
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print("\nImagini potrivite:")
    for name, sim in sorted_matches:
        print(f"{name}: Similaritate {sim:.2f}")

    return sorted_matches


if __name__ == "__main__":
    database_folder = "C:/Users/alex/PycharmProjects/PythonProject5/Images"
    target_image_path = "C:/Users/alex/PycharmProjects/PythonProject5/Test/test_image.jpeg"

    print("Analizăm baza de date...")
    database_embeddings = analyze_faces(database_folder)

    if database_embeddings:
        print(f"Comparăm {target_image_path} cu baza de date...")
        matches = match_face(target_image_path, database_embeddings)

        if not matches:
            print("Nu s-au găsit potriviri peste pragul setat.")
    else:
        print("Nu s-au găsit fețe valide în baza de date.")

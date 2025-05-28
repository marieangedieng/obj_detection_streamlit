import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import kagglehub

# Chargement du modèle YOLO
model = YOLO("yolov8_reentraine.pt")

# Téléchargement du dataset via KaggleHub
@st.cache_data
def charger_donnees():
    path = kagglehub.dataset_download("thepbordin/indoor-object-detection")

    def charger_dataset(img_dir, label_dir):
        images = []
        image_paths = []
        labels = []

        for img_name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            image_paths.append(img_path)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    boxes = []
                    for line in f:
                        parts = line.strip().split()
                        boxes.append([float(p) for p in parts])
                    labels.append(boxes)
            else:
                labels.append([])

        return images, labels, image_paths

    return charger_dataset(
        os.path.join(path, "test", "images"),
        os.path.join(path, "test", "labels")
    )

# Chargement des images et chemins
test_images, test_labels, test_image_paths = charger_donnees()

# Interface utilisateur
st.title("🧪 Tester le modèle sur une image de test")
max= len(test_images)-1
st.markdown(f"Choisissez un numéro d’image entre 0 et {max} pour visualiser la prédiction du modèle, puis cliquez ENTREE")

index = st.number_input("Index de l’image :", min_value=0, max_value= max - 1, step=1)

# Affichage image originale
image_np = test_images[index]
st.image(image_np, caption=f"Image de test {index}", use_container_width=True)

# Prédiction YOLO
img_path = test_image_paths[index]
results = model.predict(img_path, save=False)

# Affichage de l'image avec détections
result_img = Image.fromarray(results[0].plot())
st.image(result_img, caption="Résultat de la détection par YOLOv8 sans seuil", use_container_width=True)

# Affichage des objets détectés
st.markdown("### Objets détectés (sans seuil):")
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = model.names[cls]
    st.write(f"- **{label}** ({conf:.2f})")

# Application du seuil de 0.5
threshold = 0.5
filtered_boxes = [box for box in results[0].boxes if float(box.conf[0]) > threshold]

# Création d'une nouvelle image avec seulement les détections filtrées
results_thresh = results[0]
results_thresh.boxes = filtered_boxes  # on garde uniquement les boxes filtrées
result_img_thresh = Image.fromarray(results_thresh.plot())

# Affichage de l'image avec seuil
st.image(result_img_thresh, caption="Résultat de la détection par YOLOv8 avec seuil de 0.5", use_container_width=True)

# Affichage des objets détectés (avec seuil)
st.markdown("### Objets détectés (seuil > 0.5) :")
for box in filtered_boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = model.names[cls]
    st.write(f"- **{label}** ({conf:.2f})")
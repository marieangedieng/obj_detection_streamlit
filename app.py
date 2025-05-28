import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Détection d'objets - Projet", layout="wide")

st.title("🧠 Détection d'objets avec YOLOv8")
st.subheader("Projet de détection d'objets en intérieur avec YOLOv8")

st.markdown("""
Ce projet utilise **YOLOv8** pour détecter des objets courants dans des environnements intérieurs, comme des portes, des fenêtres, des chaises, etc.

Les images ont été annotées puis le modèle a été réentraîné pour s'adapter à ce contexte.

### 🔍 Classes détectées
- door
- cabinetDoor
- refrigeratorDoor
- window
- chair
- table
- cabinet
- couch
- openedDoor
- pole

### 📸 Quelques résultats

Voici quelques exemples d’images issues des prédictions du modèle :
""")

image_dir = "test_images"
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])[:3]

cols = st.columns(len(image_files))
for col, img_file in zip(cols, image_files):
    img = Image.open(os.path.join(image_dir, img_file))
    col.image(img, caption=img_file, width=400, use_container_width=False)

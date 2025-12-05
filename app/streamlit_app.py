from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Imports utilitaires internes au projet
from src.utils.utils_data import list_available_ids, load_image_and_mask
from src.utils.utils_api import send_image_to_api
from src.utils.utils_visual import colorize_mask

# ------------------------------------------------------------
# Configuration des chemins
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = Path(
    os.getenv(
        "IMAGES_DIR",
        PROJECT_ROOT / "data" / "processed" / "images" / "test",
    )
)
MASKS_DIR = Path(
    os.getenv(
        "MASKS_DIR",
        PROJECT_ROOT / "data" / "processed" / "masks" / "test",
    )
)

# URL de l’API (modifiable via variable d’environnement)
API_URL = os.getenv(
    "API_URL",
    "https://p8oc-api-6972f71da6e9.herokuapp.com/predict",
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau numpy (H, W, 3) en image PIL."""
    return Image.fromarray(arr.astype(np.uint8))


@st.cache_data
def get_available_ids():
    """Retourne la liste des IDs disponibles (mise en cache)."""
    return list_available_ids(IMAGES_DIR)


@st.cache_data
def get_image_and_mask(image_id: str):
    """Charge l’image et le masque correspondant à un ID (mis en cache)."""
    return load_image_and_mask(image_id, IMAGES_DIR, MASKS_DIR)


# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.set_page_config(
    page_title="Projet P8 - Segmentation Cityscapes",
    layout="wide",
)

st.title("🚗 Projet P8 – Segmentation de scènes urbaines")
st.markdown(
    """
Application de démonstration du modèle de segmentation entraîné sur Cityscapes.

**Workflow :**
1. Sélection d’un ID d’image.
2. Chargement de l’image RGB et du masque réel.
3. Envoi de l’image à l’API de segmentation.
4. Visualisation du masque prédit et comparaison avec le masque réel.
"""
)

# Sidebar : informations de configuration
st.sidebar.header("Configuration")
st.sidebar.write(f"📁 Dossier images : `{IMAGES_DIR}`")
st.sidebar.write(f"📁 Dossier masques : `{MASKS_DIR}`")
st.sidebar.write(f"🌐 URL API : `{API_URL}`")

# ------------------------------------------------------------
# Sélection et traitement de l'image
# ------------------------------------------------------------
try:
    ids = get_available_ids()
except Exception as e:
    st.error(f"Impossible de lister les IDs dans `{IMAGES_DIR}` : {e}")
    st.stop()

if not ids:
    st.error(f"Aucune image détectée dans `{IMAGES_DIR}`.")
    st.stop()

selected_id = st.selectbox("Sélection de l’ID de l’image :", ids)

if st.button("Lancer la prédiction sur cet ID"):

    # Chargement image + masque réel
    with st.spinner("Chargement des données..."):
        try:
            image_rgb, mask_true = get_image_and_mask(selected_id)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données pour `{selected_id}` : {e}")
            st.stop()

    # Appel API
    with st.spinner("Appel à l’API de segmentation..."):
        try:
            mask_pred = send_image_to_api(image_rgb, API_URL)
        except Exception as e:
            st.error(f"Erreur lors de l’appel API : {e}")
            st.stop()

    # Colorisation pour visualisation
    try:
        mask_true_color = colorize_mask(mask_true)
        mask_pred_color = colorize_mask(mask_pred)
    except Exception as e:
        st.error(f"Erreur lors de la colorisation des masques : {e}")
        st.stop()

    # --------------------------------------------------------
    # Affichage des résultats
    # --------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Image RGB")
        st.image(np_to_pil(image_rgb), use_column_width=True)

    with col2:
        st.subheader("Masque réel")
        st.image(np_to_pil(mask_true_color), use_column_width=True)

    with col3:
        st.subheader("Masque prédit")
        st.image(np_to_pil(mask_pred_color), use_column_width=True)

    st.success("Prédiction terminée.")
else:
    st.info("Sélectionner un ID puis lancer la prédiction.")

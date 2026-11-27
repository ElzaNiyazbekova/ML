import base64
import html
import mimetypes
from pathlib import Path
from typing import Optional
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

SPECIES = {
    "Adelie": {
        "description": "Small penguins with short bills, found around Antarctica.",
        "image_url": "/Users/elzaniyazbekova/Desktop/MLLM/SIS3/photos/Screenshot 2026-11-26 at 21.17.59.png",
    },
    "Chinstrap": {
        "description": "Recognizable by the thin black band under the head.",
        "image_url": "/Users/elzaniyazbekova/Desktop/MLLM/SIS3/photos/Screenshot 2026-11-26 at 21.18.49.png",
    },
    "Gentoo": {
        "description": "The largest of these three, with long flippers and an orange bill.",
        "image_url": "/Users/elzaniyazbekova/Desktop/MLLM/SIS3/photos/Screenshot 2026-11-26 at 21.19.29.png",
    },
}

st.set_page_config(page_title="Penguin Classifier", page_icon="🐧", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f1ff 0%, #f0f8f0 100%);
        color: #39334a;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 241, 255, 0.8);
    }
    h1, h2, h3 {
        color: #654b86;
    }
    [data-testid="stWidgetLabel"] p {
        color: #514565;
    }
    div.stButton > button {
        background: #8c70b3;
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
    }
    div.stButton > button:hover {
        background: #725796;
        color: white;
        border: 0;
    }
    .species-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #d8c9ec;
        border-radius: 18px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 5px 16px rgba(98, 75, 130, 0.09);
    }
    .species-card.predicted {
        border: 2px solid #88b99a;
        background: #f2fbf3;
    }
    .species-card h3 {
        margin: 0 0 10px;
        color: #654b86;
    }
    .species-card p {
        color: #514b5d;
        min-height: 48px;
    }
    .species-photo, .photo-placeholder {
        width: 100%;
        height: 190px;
        object-fit: cover;
        border-radius: 12px;
        background: #eee7f6;
    }
    .photo-placeholder {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #796a8d;
        text-align: center;
        border: 1px dashed #bba8d4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_embedded_image_source(image_reference: str) -> Optional[str]:
    image_reference = image_reference.strip()
    if not image_reference:
        return None
    if image_reference.startswith(("http://", "https://")):
        return image_reference

    image_path = Path(image_reference)
    if not image_path.is_absolute():
        image_path = Path(__file__).parent / image_path
    if not image_path.is_file():
        st.warning(f"Could not find species photo: {image_path}")
        return None

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        st.warning(f"Could not determine image type for: {image_path}")
        return None

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def render_species_card(name: str, details: dict[str, str], prediction: Optional[str]) -> None:
    image_source = get_embedded_image_source(details["image_url"])
    photo = (
        f'<img class="species-photo" src="{html.escape(image_source, quote=True)}" '
        f'alt="{html.escape(name, quote=True)} penguin">'
        if image_source
        else f'<div class="photo-placeholder">Set a valid photo path or URL<br>in SPECIES</div>'
    )
    card_class = "species-card predicted" if name == prediction else "species-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <h3>{html.escape(name)}</h3>
            {photo}
            <p>{html.escape(details["description"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("🐧 Penguin Species Classifier")
    st.write("Explore the three species, then enter measurements to predict one.")

    st.subheader("Enter penguin measurements")
    with st.form("penguin_measurements"):
        col1, col2 = st.columns(2)
        with col1:
            bill_length = st.number_input("Bill length (mm)", min_value=0.0)
            bill_depth = st.number_input("Bill depth (mm)", min_value=0.0)
        with col2:
            flipper_length = st.number_input("Flipper length (mm)", min_value=0.0)
            body_mass = st.number_input("Body mass (g)", min_value=0.0)
        submitted = st.form_submit_button("Predict species")

    prediction: Optional[str] = None
    if submitted:
        measurements = {
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass,
        }
        if any(value <= 0 for value in measurements.values()):
            st.warning("Enter a positive value for each measurement.")
        else:
                response = requests.post(
                    f"{API_URL.rstrip('/')}/predict",
                    json=measurements,
                    timeout=15,
                )
                response.raise_for_status()
                result = response.json()
            
                if not isinstance(result, dict) or not isinstance(
                    result.get("predicted_class_name"), str
                ):
                    st.error("The prediction API returned an unexpected response.")
                else:
                    prediction = result["predicted_class_name"]
                    st.success(f"Estimated species: {prediction}")
                    if prediction not in SPECIES:
                        st.warning(
                            f"No species photo card is configured for {prediction}."
                        )

    st.subheader("Meet the species")
    columns = st.columns(3)
    for column, (name, details) in zip(columns, SPECIES.items()):
        with column:
            render_species_card(name, details, prediction)


if __name__ == "__main__":
    main()

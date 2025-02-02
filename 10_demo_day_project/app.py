import streamlit as st
from PIL import Image
import numpy as np
import io
import requests

# FastAPI Endpoint
API_URL = "https://MSInTech/skin-dataset-project.hf.space"

# ✅ Vérification automatique de la connexion à FastAPI
try:
    response = requests.get(f"{API_URL}/health", timeout=3)
    api_status = response.json()["status"] if response.status_code == 200 else "❌ API non disponible"
except:
    api_status = "❌ API non disponible"

# Configuration de la page avec nouvelle icône
st.set_page_config(
    page_title="Skin Type Classifier",
    page_icon="🎭",
    layout="wide"
)

# Ajout d'un fond de couleur pour la sidebar et ajustements CSS
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8faff;
        }
        .sidebar-links p {
            margin-bottom: 4px;
            font-size: 14px;
        }
        .title-text {
            font-size: 40px !important;
            font-weight: bold;
            text-align: left;
        }
        .welcome-text {
            font-size: 24px !important;
            font-weight: bold;
            text-align: left;
        }
        .subheading {
            font-size: 20px !important;
            font-weight: bold;
        }
        .description {
            font-size: 18px !important;
        }
        div.stButton > button {
            background: #FFC670;
            color: black;
            font-size: 24px;
            font-weight: bold;
            padding: 15px 20px;
            border: none;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
            width: 100%;
        }
        div.stButton > button:hover {
            background: #f8faff;
            color: black;
        }
        /* ✅ Style pour les conseils de soin */
        .skin-advice {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .checklist {
            font-size: 18px;
            list-style-type: none;
            padding-left: 0;
        }
        .checklist li {
            margin-bottom: 5px;
        }
        .check-icon {
            color: #FF8C00;
            font-weight: bold;
            font-size: 22px;
            margin-right: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar - Informations et crédits
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGJ1VThlzsofnVFYiAwyQGnEXRbBAC2OTrzAa0WxKqOh1bbLPTBm-Q1LLVSRHFfkd1cd8&usqp=CAU", width=80)

    st.markdown("### Learn More About Skin Types")
    st.markdown(
        """
        <div class="sidebar-links">
            <p><a href="https://en.wikipedia.org/wiki/Oily_skin" target="_blank">Oily Skin</a></p>
            <p><a href="https://en.wikipedia.org/wiki/Dry_skin" target="_blank">Dry Skin</a></p>
            <p><a href="https://en.wikipedia.org/wiki/Normal_skin" target="_blank">Normal Skin</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Dataset Source")
    st.markdown("Kaggle: [Oily, Dry, and Normal Skin Types Dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset/data)")

    st.markdown("### About This Project")
    st.write("This project was created by [**Marie-Sophie Chenevier**](https://www.linkedin.com/in/mariesophiechenevier) as part of the [Jedha Full Stack Bootcamp](https://www.jedha.co/formations/formation-data-scientist), a high-quality Data Science training program recognized by the French government.")
    st.write("I am now looking for a meaningful job in Vaud or Geneva, Switzerland! Feel free to contact me: [mschenevier@gmail.com](mailto:mschenevier@gmail.com)")
    st.write("[GitHub](https://github.com/msche81)")

    st.markdown("**© February 2025**")

# **Titre de l'application** (aligné à gauche)
st.markdown('<p class="title-text">🎭 Skin Type Classifier</p>', unsafe_allow_html=True)

# **Ajout d'un message de bienvenue**
st.markdown('<p class="welcome-text">Welcome to this application!</p>', unsafe_allow_html=True)
st.markdown('<p class="description">This app helps you identify your skin type and provides additional care tips. Developed as part of my Data Science training, this project showcases real-world AI applications.</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Improvement of overall skin health, boost of confidence, and promotion of relaxation through self-care!</p>', unsafe_allow_html=True)

st.markdown('<p class="subheading">Upload an image of skin and get a classification of the skin type:</p>', unsafe_allow_html=True)

# **Deux colonnes : à gauche l'upload, à droite le bouton + résultats**
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=False, width=350)

    with col2:
        # ✅ **Bouton fonctionnel avec Streamlit**
        if st.button("Click here to predict your Skin Type!"):
            skin_type = np.random.choice(["Oily", "Dry", "Normal"])

            # ✅ Texte affichant le type de peau SANS le cadre vert
            st.markdown(
                f"""
                <p style='font-size:28px; font-weight:bold; text-align:center; color:#1C4C6A;'>
                    Your skin is classified as
                </p>
                <p style='font-size:38px; font-weight:bold; text-align:center; color:#FF8C00;'>
                {skin_type}
                </p>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            # ✅ Conseils spécifiques au type de peau avec checkmarks √ en orange
            if skin_type == "Oily":
                st.markdown(
                    """
                    <p class="skin-advice">🌟 Oily Skin: Produces excess sebum, leading to shine and potential acne.</p>
                    <ul class="checklist">
                        <li><span class="check-icon">√</span> Use oil-free moisturizers and gentle cleansers.</li>
                        <li><span class="check-icon">√</span> Avoid heavy creams.</li>
                        <li><span class="check-icon">√</span> Exfoliate weekly to unclog pores.</li>
                    </ul>
                    """,
                    unsafe_allow_html=True
                )
            elif skin_type == "Dry":
                st.markdown(
                    """
                    <p class="skin-advice">💧 Dry Skin: Lacks moisture, leading to flakiness and irritation.</p>
                    <ul class="checklist">
                        <li><span class="check-icon">√</span> Use hydrating creams.</li>
                        <li><span class="check-icon">√</span> Avoid harsh soaps.</li>
                        <li><span class="check-icon">√</span> Apply a moisturizing serum before bed.</li>
                    </ul>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <p class="skin-advice">😊 Normal Skin: Well-balanced, not too oily or dry.</p>
                    <ul class="checklist">
                        <li><span class="check-icon">√</span> Maintain a simple skincare routine.</li>
                        <li><span class="check-icon">√</span> Stay hydrated.</li>
                        <li><span class="check-icon">√</span> Use a daily SPF to protect your skin.</li>
                    </ul>
                    """,
                    unsafe_allow_html=True
                )

# ✅ **Section en pleine largeur après les colonnes**
st.markdown("---")
st.markdown('<p class="welcome-text">💡 What We Learned @Jedha</p>', unsafe_allow_html=True)
st.write("""
✅ Collect & Store Big Data / ✅ Build Machine Learning & Deep Learning models / ✅ Deploy models in real-world conditions""")

st.markdown('<p class="welcome-text">🛠️ Technical Skills</p>', unsafe_allow_html=True)
st.write("""
- **Programming & Tools:** Python, SQL, Tableau, Looker Studio, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow  
- **Data Collection:** Webscraping  
- **Storage & ETL:** MySQL, PostgreSQL, AWS, Google Cloud, Spark, RDD, SparkSQL  
- **Project Management:** Agile, Kanban, Jira, Confluence, Slack  
- **Data Visualization:** Looker Studio, Plotly  
- **Machine Learning & AI:**  
  - *Supervised:* Regression, Trees, Random Forest, Time Series, Model Selection  
  - *Unsupervised:* KMeans, DBSCAN, Dimensionality Reduction PCA, Natural Language Processing for Unsupervised Learning, Topic Modeling  
  - *Deep Learning:* CNNs, Transfer Learning, GANs, Word Embedding, Encoder Decoder, Attention  
- **Deployment:** Local, Web Dashboard, Docker, MLFlow, FastAPI, HuggingFace  
- **Collaboration:** Git, Bitbucket, Docker  
""")
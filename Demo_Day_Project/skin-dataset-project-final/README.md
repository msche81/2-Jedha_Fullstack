---
title: Skin Type Classifier
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: docker
app_file: main.py
pinned: false
---

# **🧑‍⚕️ Skin Type Classifier - AI-Powered Skincare Recommendation**
🚀 **Powered by MobileNetV2, FastAPI & Streamlit | Deployed on Hugging Face Spaces**  

## **🔍 Overview**
This project leverages **Deep Learning** to classify skin types (**Oily, Dry, Normal**) based on image inputs.  
The model helps users determine their skin type and provides tailored skincare recommendations.  

### **🤖 Model Architecture**
The classifier is built using **MobileNetV2**, a **pre-trained Convolutional Neural Network (CNN)** known for its efficiency in image recognition tasks.  
This transfer learning approach improves performance while keeping the model lightweight.  

It is integrated into a full-stack AI application with:
- **TensorFlow/Keras** for the MobileNetV2-based skin type classifier  
- **FastAPI** for the backend API  
- **Streamlit** for the interactive web interface  
- **Docker** for containerized deployment  

---

## **📌 Features**
✔️ **Upload an image** of your skin  
✔️ **AI-powered classification** into Oily, Dry, or Normal skin  
✔️ **Personalized skincare recommendations**  
✔️ **Lightweight CNN model for fast inference (MobileNetV2)**  
✔️ **Interactive web interface with Streamlit**  
✔️ **Deployed on Hugging Face Spaces**  

---

## **🛠️ Tech Stack**
| Technology | Purpose |
|------------|---------|
| `TensorFlow / Keras` | Deep Learning Model (MobileNetV2 for Transfer Learning) |
| `FastAPI` | Backend API for image processing & prediction |
| `Streamlit` | Frontend for user interaction |
| `Docker` | Containerization for deployment |
| `Hugging Face Spaces` | Hosting the app |

---

## **💻 How to Run Locally**
To run this project on your machine:

1️⃣ **Clone the repository**  
```bash
git clone https://huggingface.co/spaces/MSInTech/skin-dataset-project-final
cd skin-dataset-project-final

2️⃣ **Install dependencies**  
poetry install

3️⃣ **Run the FastAPI backend**  
poetry run uvicorn main:app --reload

4️⃣ **Run the FastAPI backend**  
poetry run streamlit run app.py

---

## **🚀  Deployment on Hugging Face**
This project is deployed on Hugging Face Spaces using the Docker SDK. It ensures seamless scaling and integration with ML models.

To deploy updates:

git add .
git commit -m "Updated model with MobileNetV2"
git push

The app will automatically rebuild and go live on Hugging Face.

---

📝 About

Author: Marie-Sophie Chenevier
Part of Jedha Full Stack Bootcamp 🏩
This project was developed during the Jedha Full Stack Data Science program to showcase AI applications in skincare.

---

📝 Ready to Try?

Upload your image and get your AI-powered skin type prediction instantly! 🧑‍⚕️

---
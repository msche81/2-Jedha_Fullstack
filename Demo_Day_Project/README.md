---
title: Skin Type Classifier
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: docker
app_file: main.py
pinned: false
---

### **🧑‍⚕️ Skin Type Classifier - AI-Powered Skincare Recommendation**
🚀 **Powered by FastAPI & Streamlit | Deployed on Hugging Face Spaces**

#### **🔍 Overview**
This project leverages **Deep Learning** to classify skin types (**Oily, Dry, Normal**) based on image inputs. The model helps users determine their skin type and provides tailored skincare recommendations.  

It is built using:
- **TensorFlow/Keras** for the skin type classification model  
- **FastAPI** for the backend API  
- **Streamlit** for the interactive web interface  
- **Docker** for containerized deployment  

---

### **📌 Features**
👉 **Upload an image** of your skin  
👉 **AI-powered classification** into Oily, Dry, or Normal skin  
👉 **Personalized skincare tips**  
👉 **Interactive web interface with Streamlit**  
👉 **Deployed on Hugging Face Spaces**  

---

### **🛠️ Tech Stack**
| Technology | Purpose |
|------------|---------|
| `TensorFlow` | Deep Learning Model for Skin Type Prediction |
| `FastAPI` | Backend API for image processing & prediction |
| `Streamlit` | Frontend for user interaction |
| `Docker` | Containerization for deployment |
| `Hugging Face Spaces` | Hosting the app |

---

### **💻 How to Run Locally**
To run this project locally:

1⃣ Clone the repository:
```bash
git clone https://huggingface.co/spaces/MSInTech/skin-dataset-project
cd skin-dataset-project
```
2⃣ Install dependencies:
```bash
poetry install
```
3⃣ Run FastAPI:
```bash
poetry run uvicorn main:app --reload
```
4⃣ Run Streamlit:
```bash
poetry run streamlit run app.py
```

---

### **🚀 Deployment on Hugging Face**
This project is deployed on **Hugging Face Spaces** using the **Docker SDK**. It allows seamless scaling and integration with ML models.  

To deploy your own version:
```bash
git add .
git commit -m "Initial commit"
git push
```
The app will automatically build and go live on Hugging Face.

---

### **📝 About**
**Author:** [Marie-Sophie Chenevier](https://www.linkedin.com/in/mariesophiechenevier)  
**Part of Jedha Full Stack Bootcamp 🏩**  
This project was built during the **Jedha Full Stack Data Science program** to demonstrate practical AI applications in skincare.  

🔗 **[Live Demo on Hugging Face](https://huggingface.co/spaces/MSInTech/skin-dataset-project)**  

---

### **🔧 Future Improvements**
- 🏆 **Model optimization with MLflow**  
- 🔄 **Improved preprocessing pipeline**  
- 📊 **Hyperparameter tuning for better accuracy**  
- 📊 **User feedback loop for continuous learning**  

---

## **✨ Ready to Try?**
Upload your image and **get your skin type prediction instantly!** 🧔  

---
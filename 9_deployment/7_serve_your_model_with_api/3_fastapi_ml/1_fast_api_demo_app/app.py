import mlflow 
import uvicorn
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib

description = """
Welcome to Jedha demo API. This app is made for you to understand how FastAPI works! Try it out 🕹️

## Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.
* `/greetings`: **GET** request that display a "hello message"

## Blog Endpoints

Imagine this API deals with blog articles. With the following endpoints, you can retrieve and create blog posts 
* `/blog-articles/{blog_id}`: **GET** request that retrieve a blog article given a `blog_id` as `int`.
* `/create-blog-article`: POST request that creates a new article

## Machine Learning

This is a Machine Learning endpoint that predict salary given some years of experience. Here is the endpoint:

* `/predict` that accepts `floats`


Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Simple endpoints to try out!",
    },

    {
        "name": "Blog Endpoints",
        "description": "More complex endpoints that deals with actual data with **GET** and **POST** requests."
    },

    {
        "name": "Machine Learning",
        "description": "Prediction Endpoint."
    }
]

app = FastAPI(
    title="🪐 Jedha Demo API",
    description=description,
    version="0.1",
    contact={
        "name": "Jedha",
        "url": "https://jedha.co",
    },
    openapi_tags=tags_metadata
)

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"

class PredictionFeatures(BaseModel):
    YearsExperience: float

@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message

@app.get("/greetings", tags=["Introduction Endpoints"])
async def greetings(name: str="Mr (or Miss) Nobody"):
    """
    Say hi to anybody who's specifying their name as query parameter. 
    """
    greetings = {
        "Message": f"Hello {name} How are you today?"
    }
    return greetings


@app.get("/blog-articles/{blog_id}", tags=["Blog Endpoints"])
async def read_blog_article(blog_id: int):
    """
    Say hi to anybody who's specifying their name as query parameter. 

    >👋 Careful, if you change the file using `/create-blog-article` right before, the new dataframe is not right away available, you will access a previous version. 
    """

    articles = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv")
    if blog_id > len(articles):
        response = {
            "msg": "We don't have that many articles!"
        }
    else:
        article = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv").iloc[blog_id, :]
        response = {
            "title": article.title,
            "content": article.content, 
            "author": article.author
        }

    return response


@app.post("/create-blog-article", tags=["Blog Endpoints"])
async def create_blog_article(blog_article: BlogArticles):
    """
    Append a new blog article into the database which is a `.csv` file. 
    """
    df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv")
    new_article = pd.Series({
        'id': len(df)+1,
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author
    })

    df = df.append(new_article, ignore_index=True)
    df.to_csv('s3://full-stack-assets/Deployment/articles.csv')

    return df.to_json()

@app.post("/post-picture", tags=["Blog Endpoints"])
async def post_picture(file: UploadFile= File(...)):
    """
    Upload a picture and read its file name.
    """
    return {"picture": file.filename}


@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction of salary for a given year of experience! 
    """
    # Read data 
    years_experience = pd.DataFrame({"YearsExperience": [predictionFeatures.YearsExperience]})

    # Log model from mlflow 
    logged_model = 'runs:/c09d09ef14e546b08f2f339d2c966da6/salary_estimator'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # If you want to load model persisted locally
    #loaded_model = joblib.load('salary_predictor/model.joblib')

    prediction = loaded_model.predict(years_experience)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response
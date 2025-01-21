import uvicorn
from fastapi import FastAPI

app = FastAPI()

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"

## form
import requests
payload = {
  "title": "This is my great blog title",
   "content": "This is the body of my article",
   "Author": "Jaskier"
}
r = requests.post("ENDPOINT_OF_MY_API", data=payload)

## get request examples
@app.get("/")
async def index():

    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"

    return message

@app.get("/hello")
async def hi():
    return 'Hello there 🤗'

@app.get("/custom-greetings")
async def custom_greetings(name: str = "Mr (or Miss) Nobody"):
    greetings = {
        "Message": f"Hello {name} How are you today?"
    }
    return greetings

@app.get("/blog-articles/{blog_id}")
async def read_blog_article(blog_id: int):

    articles = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv")
    if blog_id > len(articles):
        response = {
            "msg": "We don't have that many articles!"
        }
    else:
        article = articles.iloc[blog_id, :]
        response = {
            "title": article.title,
            "content": article.content, 
            "author": article.author
        }

    return response

## post request examples
@app.post("/create-blog-article")
async def create_blog_article(blog_article: BlogArticles):
    df = pd.read_csv("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv")
    new_article = pd.Series({
        'id': len(df)+1,
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author
    })

    df = df.append(new_article, ignore_index=True)

    return df.to_json()

@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author
    }
    return example_data

## Request Body
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"

@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author
    }
    return example_data


## Here the endpoint can accept either floats or ints!
from typing import Literal, List, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"
    avg_reading_time: Union[int, float] # Average reading time which can be a int or float

@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author,
        'average_reading_time': blog_article.avg_reading_time # Average reading time which can be a int or float
    }
    return example_data

## unique values between a list
from typing import Literal, List, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"
    avg_reading_time: Union[int, float]
    category: Literal["Tech", "Environment", "Politics"] = "Tech" # Literal representing a category that can be only between "Tech", "Environment", "Politics". Default is "Tech"

@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author,
        'average_reading_time': blog_article.avg_reading_time,
        'category': blog_article.category # Literal representing a category that can be only between "Tech", "Environment", "Politics". Default is "Tech"
    }
    return example_data

# accept list
from typing import Literal, List, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

class BlogArticles(BaseModel):
    title: str
    content: str
    author: str = "Anonymous Author"
    avg_reading_time: Union[int, float]
    category: Literal["Tech", "Environment", "Politics"] = "Tech"
    tags: List[str] = None # Accepts only a list of strings, and default is None (meaning nothing)

@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        'title': blog_article.title,
        'content': blog_article.content,
        'author': blog_article.author,
        'average_reading_time': blog_article.avg_reading_time, 
        'category': blog_article.category,
        'tags': blog_article.tags # Accepts only a list of strings, and default is None (meaning nothing)
    }
    return example_data

## Basic app metadata
description = """
This is your app description, written in markdown code

# This is a title

* This is a bullet point
"""

tag_metadata = [
    {
        "name": "Name_1",
        "description": "LOREM IPSUM NEC."
    },

    {
        "name": "Name_2",
        "description": "LOREM IPSUM NEC."
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

@app.get("/", tags=["Name_1"]) # here we categorized this endpoint as part of "Name_1" tag
async def index():
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message

## Document each endpoint

@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message
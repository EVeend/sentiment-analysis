from fastapi import FastAPI
from routers import sentiment_analysis_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

app.include_router(sentiment_analysis_router.router, prefix="/model")
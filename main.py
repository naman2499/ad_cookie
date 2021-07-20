from typing import Optional
from fastapi import FastAPI
import main_combine

app = FastAPI()

@app.get("/")
def read_root(text):
    print(text)
    return main_combine.main(text)


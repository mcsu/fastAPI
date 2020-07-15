from typing import Optional

from fastapi import FastAPI, Query
from app import run

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "ng"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}


@app.post("/{base64data}")
def call_predict(base64data: str):
    result = run(base64data)
    return {"result": result}


@app.get("/items/{regex}")
def regex(q: Optional[str] = Query(None, regex="^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$")):
    return run(q)

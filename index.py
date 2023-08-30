"""
Exemple file for starting a web server in a datacage and answering to requests
"""
import asyncio
import os
import pathlib
import signal
import sys
import typing
from typing import Any, TypedDict

import uvicorn
from fastapi import (
    APIRouter,
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from dv_utils import default_settings, Client, audit_log

##################

APP_ID = os.environ.get("DV_CAGE_ID",None)
ROOT_PATH = ""
if "DV_CAGE_ID" in os.environ:
    ROOT_PATH = "/" + os.environ["DV_CAGE_ID"].split("-")[-1]
else:
    #    ROOT_PATH = "/CAGE_ID"
    ROOT_PATH = ""
print("ROOT_PATH=", ROOT_PATH)

app = FastAPI(
    title="webserver-in-cage",
    description="""
        datavillage template of a web server running in a datacage
    """,
    root_path=ROOT_PATH,
)
app_router = APIRouter(prefix=ROOT_PATH)


@app.get("/")
async def root():
    """
    Route endpoint return an "hello world" response
    """
    return {"message": f"Hello world from a data cage"}


@app.post("/log")
async def push_log(log: str = "empty log"):
    """
    Example of a route receiving a log string and pushing it to the log queue
    """
    audit_log(APP_ID, f"upload_data {in_file.filename}")
    return {"message": f"The log '{log}' was properly pushed to the log stack"}

@app.get("/users")
async def list_users():
    """
    Route endpoint that return the list of users connected to the collaboration space
    """
    client = Client()
    user_ids = client.get_users()
    
    return {"user_ids": user_ids}


async def web_server():
    """
    Start the web server
    """
    audit_log(APP_ID, "web server starting")

    config = uvicorn.Config(app, port=80, host="0.0.0.0", reload=True)
    server = uvicorn.Server(config)
    await server.serve()

    audit_log(APP_ID, "web server closing")


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except Exception:  # pylint: disable=broad-except
        loop = asyncio.new_event_loop()
    loop.run_until_complete(web_server())

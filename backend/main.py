from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("../shared/questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

sessions = {}  # {session_id: state}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.get_template("index.html").render({"request": request})

@app.post("/start")
def start_interview(session_id: str, type: str):
    if type == "technical":
        q = random.choice(QUESTIONS["easy"])
        sessions[session_id] = {
            "type": "technical",
            "questions": [q, random.choice(QUESTIONS["medium"])],
            "current_idx": 0,
            "answers": []
        }
        return {"question": q["title"], "desc": q["desc"]}
    else:
        sessions[session_id] = {
            "type": "behavioral",
            "step": "intro",
            "answers": {}
        }
        return {"question": "Tell me about yourself."}

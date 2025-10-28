from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import random
import requests
import json
from typing import Literal
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llm import ask_llm  # Import ask_llm from llm.py

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load questions
with open("../shared/questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

sessions = {}
model = SentenceTransformer('all-MiniLM-L6-v2')
behavioral_qs = QUESTIONS["behavioral"]
embs = model.encode(behavioral_qs)
index = faiss.IndexFlatL2(384)
index.add(np.array(embs))

class StartRequest(BaseModel):
    session_id: str
    type: Literal["technical", "behavioral"]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.get_template("index.html").render({"request": request})

@app.post("/start")
def start_interview(request: StartRequest):
    if request.type == "technical":
        q = random.choice(QUESTIONS["easy"])
        sessions[request.session_id] = {
            "type": "technical",
            "questions": [q, random.choice(QUESTIONS["medium"])],
            "current_idx": 0,
            "answers": []
        }
        return {"question": q["title"], "desc": q["desc"]}
    else:
        sessions[request.session_id] = {
            "type": "behavioral",
            "step": "intro",
            "answers": {}
        }
        return {"question": "Tell me about yourself."}

class CodeSubmit(BaseModel):
    session_id: str
    code: str
    language: str = "python"

LANGUAGES = {"python": 71, "javascript": 63}

@app.post("/submit_code")
def submit_code(data: CodeSubmit):
    sess = sessions.get(data.session_id)
    if not sess or sess["type"] != "technical":
        return {"error": "Invalid or non-technical session"}

    # Get test cases for the current question
    current_question = sess["questions"][sess["current_idx"]]
    if not current_question.get("test_cases"):
        return {"error": "No test cases found for the current question"}

    lang_id = LANGUAGES.get(data.language.lower(), 71)
    test_results = []

    # Iterate through all test cases
    for test_case in current_question["test_cases"]:
        stdin = test_case["input"]  # e.g., "2 7 11 15\n9"
        expected_output = test_case["output"]  # e.g., "0 1"

        # Wrap code to handle input for Two Sum
        wrapped_code = f"""
{data.code}
# Read input
nums = list(map(int, input().split()))
target = int(input())
print(twoSum(nums, target))
"""

        payload = {
            "source_code": wrapped_code,
            "language_id": lang_id,
            "stdin": stdin
        }

        try:
            r = requests.post("http://localhost:2358/submissions?wait=true", json=payload)
            r.raise_for_status()
            result = r.json()
        except requests.RequestException as e:
            return {"error": f"Judge0 request failed for test case: {str(e)}"}

        stdout = result.get("stdout")
        time = result.get("time", 0)
        memory = result.get("memory", 0)
        stderr = result.get("stderr")
        compile_error = result.get("compile_output")

        # Handle errors
        if stderr or compile_error:
            test_results.append({
                "input": stdin,
                "stdout": stdout,
                "stderr": stderr,
                "compile_error": compile_error,
                "time": time,
                "memory": memory,
                "is_correct": False
            })
            continue

        # Normalize stdout for comparison (e.g., "[0, 1]" -> "0 1")
        normalized_stdout = stdout.strip() if stdout else ""
        if normalized_stdout.startswith("[") and normalized_stdout.endswith("]"):
            try:
                parsed = json.loads(normalized_stdout)
                if isinstance(parsed, list):
                    normalized_stdout = " ".join(str(x) for x in parsed)
                else:
                    normalized_stdout = ""
            except json.JSONDecodeError:
                normalized_stdout = ""

        # Validate output
        is_correct = normalized_stdout == expected_output.strip() if normalized_stdout else False

        test_results.append({
            "input": stdin,
            "stdout": stdout,
            "expected_output": expected_output,
            "is_correct": is_correct,
            "time": time,
            "memory": memory
        })

    # LLM feedback for all test cases
    prompt = (
        f"Review this {data.language} code for a mock interview:\n{data.code}\n"
        f"Test Case Results:\n"
    )
    all_correct = all(result["is_correct"] for result in test_results)
    for i, result in enumerate(test_results, 1):
        prompt += (
            f"Test Case {i}:\n"
            f"Input: {result['input']}\n"
            f"Output: {result['stdout'] if result['stdout'] else 'None'}\n"
            f"Expected Output: {result['expected_output']}\n"
            f"Correct: {result['is_correct']}\n"
            f"Time: {result['time']}s, Memory: {result['memory']}KB\n"
        )
        if "stderr" in result:
            prompt += f"Error: {result['stderr'] or result['compile_error']}\n"
    prompt += "Provide detailed feedback on correctness, efficiency, and suggestions for improvement."
    feedback = ask_llm(prompt)

    # Store answer
    sess["answers"].append({
        "code": data.code,
        "test_results": test_results,
        "all_correct": all_correct
    })

    return {
        "test_results": test_results,
        "all_correct": all_correct,
        "feedback": feedback
    }

@app.post("/next")
def next_question(session_id: str):
    sess = sessions[session_id]
    if sess["current_idx"] < 1:
        sess["current_idx"] += 1
        q = sess["questions"][sess["current_idx"]]
        return {"question": q["title"], "desc": q["desc"]}
    else:
        answers = sess.get("answers", [])
        prompt = "Give detailed feedback on both problems. Strengths, weaknesses, improvements.\n"
        for i, ans in enumerate(answers, 1):
            prompt += f"Problem {i} Code:\n{ans['code']}\n"
            prompt += "Test Case Results:\n"
            for j, result in enumerate(ans["test_results"], 1):
                prompt += (
                    f"Test Case {j}:\n"
                    f"Output: {result['stdout'] if result['stdout'] else 'None'}\n"
                    f"Correct: {result['is_correct']}\n"
                    f"Time: {result['time']}s, Memory: {result['memory']}KB\n"
                )
        return {"feedback": ask_llm(prompt), "done": True}

@app.post("/answer_intro")
def answer_intro(session_id: str, answer: str):
    sess = sessions[session_id]
    user_emb = model.encode([answer])
    D, I = index.search(user_emb, 4)
    selected = [behavioral_qs[i] for i in I[0]]
    sess["selected"] = selected
    sess["step"] = "questions"
    sess["answers"]["intro"] = answer
    return {"next_question": selected[0]}

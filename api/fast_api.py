from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import inference


class UserInput(BaseModel):
    gravity: float
    ph: float
    osmo: float
    cond: float
    urea: float
    calc: float


app = FastAPI()


@app.post("/myapp")
def get_results(input: UserInput):
    result = inference(input.gravity, input.ph, input.osmo, input.cond, input.urea, input.calc)
    if result == 0:
        return "No kidney Stones ✅"
    else:
        return "There are kidney stone 🫨"

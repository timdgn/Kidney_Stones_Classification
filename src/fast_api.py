from fastapi import FastAPI
from pydantic import BaseModel
from inference import inference


class UserInput(BaseModel):
    gravity: float
    ph: float
    osmo: float
    cond: float
    urea: float
    calc: float


app = FastAPI()


@app.post("/inference")
def get_results(input: UserInput):
    result = inference(input.gravity, input.ph, input.osmo, input.cond, input.urea, input.calc)
    return result

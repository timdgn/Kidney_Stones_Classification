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


@app.post("/myapp")
def get_results(input: UserInput):
    """
    This function takes a UserInput object as input and returns a string indicating whether there are kidney stones or not.

    Parameters
    ----------
    input (UserInput)
        A UserInput object with six attributes: gravity, ph, osmo, cond, urea, and calc.

    Returns
    -------
    str
        A string that says "No kidney stones ✅" if the result is 0, or "There are kidney stone 🫨" if the result is 1.
    """
    result = inference(input.gravity, input.ph, input.osmo, input.cond, input.urea, input.calc)
    if result == 0:
        return "No kidney stones ✅"
    else:
        return "There are kidney stone 🫨"

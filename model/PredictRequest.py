from pydantic import BaseModel

class PredictRequest(BaseModel):
    input_string: list
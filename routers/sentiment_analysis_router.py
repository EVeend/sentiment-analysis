import pandas as pd

from io import BytesIO
from fastapi import APIRouter, HTTPException, UploadFile, File, Response

from service.sentiment_analysis_service import SentimentAnalysis
from model.PredictRequest import PredictRequest

router = APIRouter()
sentiment_analysis_service = SentimentAnalysis()

@router.post("/predict")
async def predict(input_list: PredictRequest):
    try:
        result = sentiment_analysis_service.predict(input_list.input_string)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in prediction")
    
@router.post("/predict-csv")
async def predict_csv(csv_file: UploadFile = File(...)):
    try:

        csv_content = await csv_file.read()

        dataframe = pd.read_csv(BytesIO(csv_content))

        result = sentiment_analysis_service.predict_csv(dataframe)
        csv_content = result.to_csv(index=False)
        response = Response(content=csv_content, media_type="text/csv")
        response.headers["Content-Disposition"] = 'attachment; filename="result.csv"'
        return response

    except Exception as e:
        return {"error": str(e)}
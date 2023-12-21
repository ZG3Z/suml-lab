import uvicorn
from fastapi import FastAPI
from models.passenger import Passenger
from libs.model import predict

app = FastAPI()

@app.post("/predict")
async def predict_survival(passenger: Passenger):
    data_for_prediction = [[
        passenger.Pclass,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        passenger.Embarked,
        passenger.Male,
    ]]

    result = predict(data_for_prediction, 'ml_models/model.pkl')
    return {"prediction": result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
import uvicorn
from fastapi import FastAPI, HTTPException
from models.passenger import Passenger
from libs.model import map_labels_to_numbers, predict

app = FastAPI()

@app.post("/predict")
async def predict_survival(passenger: Passenger):
    pclass = map_labels_to_numbers(passenger.Pclass, {'First': 0, 'Second': 1, 'Third': 2})
    embarked = map_labels_to_numbers(passenger.Embarked, {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2})
    sex = map_labels_to_numbers(passenger.Male, {'Female': 0, 'Male': 1})

    if any(value is None for value in [pclass, embarked, sex]):
        raise HTTPException(status_code=400, detail="Invalid input values")

    data_for_prediction = [[
        pclass,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        embarked,
        sex,
    ]]

    result = predict(data_for_prediction, 'ml_models/model.pkl')
    return {"prediction": result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
from pydantic import BaseModel

class Passenger(BaseModel):
    Pclass: int
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int
    Male: int
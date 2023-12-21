from pydantic import BaseModel

class Passenger(BaseModel):
    Pclass: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Male: str
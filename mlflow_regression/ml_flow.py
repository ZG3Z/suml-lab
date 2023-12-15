import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data/10_points.csv')

X = data[['x']]
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mlflow.set_experiment('Regression Experiment')

with mlflow.start_run():
    mlflow.log_metrics({'mse': mse, 'r2': r2})

    mlflow.set_tag('Train info', 'Linear Regression for 10_points')

    signature = infer_signature(X_train, y_train)
    model_info_reg = mlflow.sklearn.log_model(
        sk_model=linear_reg,
        artifact_path='regression_model',
        signature=signature,
        input_example=X_train,
        registered_model_name='regression_ml_model'
    )

loaded_model = mlflow.pyfunc.load_model(model_info_reg.model_uri)

predictions = loaded_model.predict(X_test)
print(predictions)

import pickle

def predict(data, model_file):
    model = pickle.load(open(model_file, 'rb'))
    survival = model.predict(data)[0]
    return 'Survived' if bool(survival) else 'Not Survived'

import pickle

def map_labels_to_numbers(value: str, label_mapping: dict) -> int:
    return label_mapping.get(value.title(), None)

def predict(data, model_file):
    model = pickle.load(open(model_file, 'rb'))
    survival = model.predict(data)[0]
    return 'Survived' if bool(survival) else 'Not Survived'

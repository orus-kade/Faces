import numpy as np


def predict_age(age_model, embedding):
    age = age_model.predict(np.array(embedding))
    return age

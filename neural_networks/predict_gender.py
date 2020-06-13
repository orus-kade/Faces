def predict_gender(gender_model, embedding):
    gender = gender_model.predict(embedding)
    gender = (gender > 0.5).astype(int)
    return gender

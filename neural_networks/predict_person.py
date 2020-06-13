def predict_person(person_model, embedding):
    person_embedding = person_model.predict(embedding)
    return person_embedding

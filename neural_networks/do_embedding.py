def do_embedding(embedding_model, face):
    embedding = embedding_model.predict(face)
    return embedding

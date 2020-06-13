from keras.models import load_model


def do_nearly_embedding(face):
    embedding_model = load_model('models/MODEL_embedding.h5')
    nearly_embedding = []
    return nearly_embedding

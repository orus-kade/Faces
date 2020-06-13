from annoy import AnnoyIndex


def write_annoy(embeddings, indices):
    f = 64  # Embedding size
    t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
    for index, emb in zip(indices, embeddings):
        t.add_item(index, emb)

    t.build(10)  # 10 trees
    t.save('./neural_networks/models/saved_annoy.ann')
from annoy import AnnoyIndex


def write_annoy(embeddings):
    f = 64  # Embedding size
    t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
    for i in range(len(embeddings)):
        t.add_item(i, embeddings[i])

    t.build(10)  # 10 trees
    t.save('./neural_networks/models/saved_annoy.ann')

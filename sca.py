import pickle
import scann
import numpy as np

print("bat dau")

vectors = pickle.load(open("vectors_resnet.pkl", "rb"))
vectors = np.array(vectors)

searcher = scann.scann_ops_pybind.builder(vectors, 50, "squared_l2").tree(
    num_leaves=89, num_leaves_to_search=20).score_brute_force(2).reorder(7).build()

searcher.serialize("indexing")

print("xong")
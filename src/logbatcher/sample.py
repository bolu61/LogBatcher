import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def dpp_sample(similarity, k):
    # similarity: similarity matrix
    # k: number of items to sample
    n = similarity.shape[0]

    # Initialize empty set Y
    Y = set()
    for _ in range(k):
        best_i = -1
        best_p = -1

        for i in range(n):
            if i not in Y:
                # Compute determinant of submatrix
                det_Yi = np.linalg.det(similarity[np.ix_(list(Y) + [i], list(Y) + [i])])

                # Compute probability of adding i to Y
                p_add = det_Yi / (1 + det_Yi)

                if p_add > best_p:
                    best_p = p_add
                    best_i = i

        # Add best item to Y
        Y.add(best_i)

    return list(Y)


def sample(strings: list[str], n=5, method="dpp") -> list[str]:
    if len(strings) <= n:
        return [*strings]

    match method:
        case "dpp":
            vectors = TfidfVectorizer().fit_transform(strings).to_array()
            similarity = cosine_similarity(vectors)
            return [strings[i] for i in dpp_sample(similarity, n)]
        case "random":
            return random.sample(strings, n)
        case _:
            raise NotImplementedError(f"{method=} not implemented")

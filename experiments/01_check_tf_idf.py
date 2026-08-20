from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.env.generator import (
    EnvironmentConfig,
    SituationGenerator,
)

def run(mismatch: float, n: int = 10_000) -> None:
    config = EnvironmentConfig(
        mismatch=mismatch,
    )

    generator = SituationGenerator(
        config=config,
        seed=67,
    )
    texts = []
    clusters = []
    decisions = []
    for _ in range(n):
        situation = generator.generate()
        texts.append(situation.text)
        clusters.append(situation.semantic_cluster)
        decisions.append(situation.decision_id)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    print(f"\nMismatch = {mismatch}")
    correct_clusters,correct_decisions = 0,0
    sim_matrix = cosine_similarity(matrix,matrix)
    for index in range(n):
        nearest_index = np.argsort(sim_matrix[index])[::-1][1]
        if clusters[index] == clusters[nearest_index]:
            correct_clusters += 1
        if decisions[index] == decisions[nearest_index]:
            correct_decisions += 1
    cluster_accuracy = correct_clusters / n
    decisions_accuracy = correct_decisions / n
    print(f"cluster_accuracy: {cluster_accuracy}")
    print(f"decisions_accuracy: {decisions_accuracy}")

if __name__ == "__main__":
    for mismatch in (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        run(mismatch)
from collections.abc import Generator, Hashable, Iterable

from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from logbatcher.template import Template


class Cluster[T: Hashable]:
    __slots__ = ["indexes"]
    indexes: dict[T, int]

    def __init__(self, items: Iterable[tuple[int, T]] | None = None):
        if items is not None:
            self.indexes = {x: i for i, x in items}

    def items(self) -> Iterable[T]:
        return self.indexes.keys()

    def index(self, item: T) -> int:
        return self.indexes[item]

    def add(self, i: int, item: T):
        self.indexes[item] = i

    def remove(self, item: T):
        del self.indexes[item]

    def __iter__(self) -> Generator[tuple[int, T]]:
        for item, i in self.indexes.items():
            yield i, item


def prune_from_cluster(template: Template, cluster: Cluster[str]):
    new_cluster = Cluster()
    for i, log in cluster:
        if template.extract(log) is not None:
            cluster.remove(log)
            cluster.add(i, log)
    return cluster, new_cluster


def vectorize(strings: Iterable[str]):
    return TfidfVectorizer().fit_transform(strings)


def cluster(strings: Iterable[str], eps=0.5):
    cluster = DBSCAN(eps=eps, min_samples=5)
    return cluster.fit_predict(vectorize(strings))


def reassign_clusters(labels: list[int], tokenized_logs: list[list[str]]) -> list[int]:
    cluster_nums = len(labels)
    mergerd_logs = []
    for tokenized_log in tokenized_logs:
        mergerd_logs.append(" ".join(tokenized_log))

    for i in range(len(labels)):
        if labels[i] == -1:
            for j in range(i + 1, len(labels)):
                if labels[j] == -1 and mergerd_logs[i] == mergerd_logs[j]:
                    labels[j] = cluster_nums
            labels[i] = cluster_nums
            cluster_nums += 1
    return labels

import calendar
import heapq
import random
import re
from collections import Counter, OrderedDict

from sklearn.cluster import DBSCAN, MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logbatcher.sample import dpp_sample, group_samples_clustering


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = (
            [
                "a",
                "an",
                "and",
                "i",
                "ie",
                "so",
                "to",
                "the",
            ]
            + list(calendar.day_name)
            + list(calendar.day_abbr)
            + list(calendar.month_name)
            + list(calendar.month_abbr)
        )
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            # print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def clean(s):
    log_format = re.sub(r"[0-9A-Za-z, ]+", "", s)
    unique_chars = list(set(log_format))
    sorted_string = "".join(sorted(unique_chars))
    s = re.sub(r':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.?!', " ", s)
    s = " ".join(
        [word for word in s.strip().split() if not bool(re.search(r"\d", word))]
    )
    return s, sorted_string


def h_clustering(contents):
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])

    # hierichical clustering
    hierichical_clusters = {}
    for k, v in contents.items():
        frequent_token = tuple(sorted(vocab.topk_tokens(v[0].split(), 3)))
        log_format = v[1]
        if frequent_token not in hierichical_clusters:
            hierichical_clusters[frequent_token] = {
                "size": 1,
                "cluster": {log_format: [k]},
            }
        else:
            hierichical_clusters[frequent_token]["size"] = (
                hierichical_clusters[frequent_token]["size"] + 1
            )
            if log_format not in hierichical_clusters[frequent_token]["cluster"]:
                hierichical_clusters[frequent_token]["cluster"][log_format] = [k]
            else:
                hierichical_clusters[frequent_token]["cluster"][log_format].append(k)
    total_coarse_clusters = len(hierichical_clusters.keys())
    total_fine_clusters = 0
    for k, v in hierichical_clusters.items():
        total_fine_clusters += len(hierichical_clusters[k]["cluster"])
    return hierichical_clusters, total_coarse_clusters, total_fine_clusters


def assign_labels(clusters, logs, granularity="coarse"):
    # Initialize the labels list with -1 for all logs
    labels = [-1] * len(logs)

    # Map each log ID to its cluster ID
    cluster_id = 0
    for frequent_tokens, cluster_info in clusters.items():
        if granularity == "coarse":
            # Assign cluster ID based on frequent tokens
            for log_format, log_ids in cluster_info["cluster"].items():
                for log_id in log_ids:
                    labels[log_id] = cluster_id
            cluster_id += 1
        elif granularity == "fine":
            # Assign unique cluster ID for each log format within frequent tokens
            for log_format, log_ids in cluster_info["cluster"].items():
                for log_id in log_ids:
                    labels[log_id] = cluster_id
                cluster_id += 1

    return labels


def hierichical_clustering(logs, granularity="fine"):
    contents = {}
    for i, x in enumerate(logs):
        x, fx = clean(x)
        if len(x.split()) > 1:
            contents[i] = (x, fx)
    clusters, a, b = h_clustering(contents)
    labels = assign_labels(clusters, logs, granularity)
    if granularity == "coarse":
        return labels, a
    else:
        return labels, b


def replace_numbers_with_zero(text):
    return re.sub(r"\d+(\.\d+)?", "0", text)


def meanshift_clustering(logs):
    text_column = [replace_numbers_with_zero(log) for log in logs]

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer()
    data_matrix = vectorizer.fit_transform(text_column).toarray()

    # Mean Shift clustering
    mean_shift = MeanShift(bandwidth=0.5)
    labels = mean_shift.fit_predict(data_matrix).tolist()
    return labels, max(labels) + 1


class Cluster:
    def __init__(self):
        self.logs = []
        self.batch_logs = []
        self.indexs = []
        self.size = 0
        self.sample_log = ""

    def append_log(self, log, index):
        self.logs.append(log)
        self.indexs.append(index)
        self.size += 1

    def varaible_sampling(self, batch_size=5, sample_method="dpp"):
        self.batch_logs = list(OrderedDict.fromkeys(self.logs))  # remove duplicates

        def _replacer(match):
            char = match.group()
            return "0" if char.isdigit() else "a"

        vars = []
        for var in self.batch_logs:
            vars.append(re.sub(r"[0-9a-zA-Z]", _replacer, var))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(vars)
        tfidf_matrix = tfidf_matrix.toarray()

        # sample
        if len(self.batch_logs) <= batch_size:
            result = range(len(self.batch_logs))
        elif sample_method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            result = dpp_sample(similarity_matrix, batch_size)
        elif sample_method == "random":
            random.seed(0)
            result = random.sample(range(0, len(self.batch_logs)), batch_size)
        elif sample_method == "similar":
            result = group_samples_clustering(tfidf_matrix, batch_size)[0]
        else:
            raise ValueError("Invalid sample method")
        self.batch_logs = [self.batch_logs[i] for i in result]

    def batching(self, batch_size=10, min_size=3, sample_method="dpp"):
        self.batch_logs = list(OrderedDict.fromkeys(self.logs))  # remove duplicates
        if len(self.batch_logs) > batch_size:
            self.sample(batch_size, sample_method)
        if type(self.batch_logs) is str:
            self.batch_logs = [self.batch_logs]
        self.sample_log = self.batch_logs[0]
        if not_varibility(self.batch_logs):
            self.batch_logs = (
                self.batch_logs[:min_size]
                if len(self.batch_logs) > min_size
                else self.batch_logs
            )

    def sample(self, batch_size, sample_method):
        # vetorize logs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.batch_logs)
        tfidf_matrix = tfidf_matrix.toarray()

        # sample
        if sample_method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            result = dpp_sample(similarity_matrix, batch_size)
        elif sample_method == "random":
            random.seed(0)
            result = random.sample(range(0, len(self.batch_logs)), batch_size)
        elif sample_method == "similar":
            result = group_samples_clustering(tfidf_matrix, batch_size)[0]
        else:
            raise ValueError("Invalid sample method")
        self.batch_logs = [self.batch_logs[i] for i in result]
        return


def tokenize(log_content, tokenize_pattern=r"[ ,|]", removeDight=True):
    words = re.split(tokenize_pattern, log_content)
    new_words = []
    for word in words:
        if "=" in word:
            ws = word.split("=")
            if len(ws) <= 2:
                new_words.append(ws[0])
            else:
                # might be some parameters of a URL
                pass

        elif removeDight and re.search(r"\d", word):
            pass
        elif "/" in word.lower() or re.match(r"^[a-zA-Z][+-]$|^[+-][a-zA-Z]$", word):
            pass
        else:
            word = re.sub(r"\([^)]*\)", "", word)
            new_words.append(word)
    new_words = [word for word in new_words if word]  # remove null
    if new_words == []:
        new_words.append(re.sub(r"\d+(\.\d+)?", "0", log_content))
    return new_words


def vectorize(tokenized_logs):
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x, lowercase=False, token_pattern=None
    )
    return vectorizer.fit_transform(tokenized_logs)


def not_varibility(logs):
    a_logs = [re.sub(r"\d+", "", log) for log in logs]
    if len(set(a_logs)) == 1:
        return True
    return False


def cluster(vectorized_logs, eps=0.5):
    cluster = DBSCAN(eps=eps, min_samples=5)
    cluster.fit(vectorized_logs)
    labels = cluster.labels_
    cluster_nums = max(labels) + 1
    return labels, cluster_nums


def reassign_clusters(labels, cluster_nums, tokenized_logs):
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
    return labels, cluster_nums


def process_new_cluster(new_cluster, clusters, batch_size, min_size=3):
    if new_cluster.size != 0:
        new_cluster.batching(batch_size, min_size)
        clusters.append(new_cluster)
        return 1
    return 0

import csv
import logging
import random
import re
from collections.abc import Sequence
from itertools import batched, islice
from typing import Any

from openai import OpenAI
from typeguard import check_type

from logbatcher.cache import ParsingCache
from logbatcher.template import Template
from logbatcher.cluster import (
    Cluster,
    cluster,
    reassign_clusters,
    vectorize,
    prune_from_cluster,
)
from logbatcher.postprocess import correct_single_template, post_process
from logbatcher.sample import sample

logger = logging.getLogger(__name__)


class LogBatcher:
    def __init__(self, /, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.model = model
        self.client = OpenAI(base_url=base_url)
        self.cache = ParsingCache()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.client.base_url,
            "cache": self.cache,
        }

    def __setstate__(self, state: dict[str, Any]):
        self.model = state["model"]
        self.client = OpenAI(base_url=state["base_url"])
        self.cache = state["cache"]

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            timeout=10,
            max_tokens=2048,
        )
        return (response.choices[0].message.content or "").strip("\n")

    # TODO: refactor
    def make_template(self, cluster: Cluster[str]) -> Template:
        # initialize
        logs = [log for i, log in cluster]

        # TODO: refactor and move to parse
        # Matching and Pruning
        for _, log in cluster:
            if (match := self.cache.match(log)) is not None:
                cluster, _ = prune_from_cluster(match.template, cluster)

        variables = sample(list(self.cache.variable_candidates), 5)

        variable_prompt = (
            f" Historical variables: {variables}." if variables != [] else ""
        )
        instruction = (
            "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name']."
            + variable_prompt
            + " Constant text and strings should not be recognized as variables.\nPrint the input log's template delimited by backticks."
        )

        # invoke LLM
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": "\n".join(
                    f"Log[{i + 1}]: `{log}`" for i, log in enumerate(logs)
                ),
            },
        ]
        logger.debug(f"{messages=}")
        answer: str
        try:
            answer = self.chat(messages)
            logger.debug(f"{answer=}")
        except Exception as error:
            logger.debug(f"while invoking llm with {messages=} got {error=}")
            answer = random.sample(logs, k=1)[0]

        template: Template = Template.from_str(post_process(answer))

        # TODO: refactor
        if not verify_template(template):
            template = correct_single_template((sample_log))

        cluster, new_cluster = prune_from_cluster(template, cluster)
        if new_cluster.size == cluster.size:
            cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
            new_cluster = Cluster()
            template = correct_single_template(tokenize(sample_log))

        return template, cluster, new_cluster

    def parse(
        self,
        logs: Sequence[str],
        /,
        batch_size: int = 10,
        chunk_size: int = 10_000,
        clustering_method: str = "dbscan",
    ) -> list[str]:
        if len(logs) <= chunk_size:
            logger.warning(f"{len(logs)=} is smaller than 10_000")

        log_chunk = []
        log_chunk_index = []
        caching = self.cache

        outputs: list[Template | None] = [None for _ in range(len(logs))]

        # Parsing
        for batch in batched(enumerate(logs), n=chunk_size):
            for i, log in batch:
                if (match := caching.match(log)) is not None:
                    outputs[i] = match.template
                else:
                    log_chunk.append(log)
                    log_chunk_index.append(i)

            # parsing start
            if clustering_method == "dbscan":
                # tokenize -> vectorize -> cluster -> reassign_clusters
                tokenized_logs = [tokenize(log) for log in log_chunk]
                labels, cluster_nums = cluster(vectorize(tokenized_logs), eps=0.5)
                labels, cluster_nums = reassign_clusters(
                    labels, cluster_nums, tokenized_logs
                )
            elif clustering_method == "hierarchical":
                labels, cluster_nums = hierichical_clustering(log_chunk)
            elif clustering_method == "meanshift":
                labels, cluster_nums = meanshift_clustering(log_chunk)
            else:
                raise ValueError("Invalid clustering method")

            # create clusters
            clusters = [Cluster() for _ in range(cluster_nums)]
            for label, log, i in zip(labels, log_chunk, log_chunk_index):
                clusters[label].append_log(log, i)

            # sorting
            clusters.sort(key=lambda cluster: len(cluster.logs), reverse=True)

            # batching
            [cluster.batching(batch_size) for cluster in clusters]

            # parsing
            for i, old_cluster in enumerate(clusters):
                template, old_cluster, new_cluster = self.make_template(old_cluster)
                # update clusters
                cluster_nums += process_new_cluster(new_cluster, clusters, batch_size)
                if template not in caching:
                    try:
                        caching.insert(
                            template=template,
                        )
                    except ValueError as e:
                        logger.warning(e)
                        continue
                    caching.variable_candidates.update(
                        vars_update(
                            old_cluster.logs[0], template, caching.variable_candidates
                        )
                    )
                for i in old_cluster.indexs:
                    outputs[i] = template

        return [check_type(s, str) for s in outputs]


def verify_template(template: Template) -> bool:
    for token in template.tokens:
        if re.search(r"\w", token) is not None:
            return True
    return False


# TODO: refactor
def vars_update(refer_log: str, template: Template, candidates: set[str]) -> list[str]:
    new_variables = extract_variables(refer_log, template)
    extend_vars = []
    if not new_variables:
        return extend_vars
    for var in new_variables:
        var = re.sub(r"^\((.*)\)$|^\[(.*)\]$", r"\1\2", var)
        if (
            var not in candidates
            and not var.isdigit()
            and not var.isalpha()
            and len(var.split()) <= 3
        ):
            extend_vars.append(var)
    return extend_vars


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logs = []
    with open("HDFS_2k.csv") as f:
        for row in islice(csv.reader(f), 1, None):
            logs.append(row[6])

    parser = LogBatcher()

    out = parser.parse(logs)

    pairs = [*zip(logs, out)]

    for pair in pairs[:10]:
        print(pair)

import csv
import logging
from collections.abc import Sequence
from itertools import batched, islice
from typing import Any

from openai import APITimeoutError, OpenAI
from typeguard import check_type

from logbatcher.additional_cluster import hierichical_clustering, meanshift_clustering
from logbatcher.cluster import (
    Cluster,
    cluster,
    process_new_cluster,
    reassign_clusters,
    tokenize,
    vectorize,
)
from logbatcher.matching import prune_from_cluster
from logbatcher.parsing_cache import ParsingCache
from logbatcher.postprocess import correct_single_template, post_process
from logbatcher.util import verify_template
from logbatcher.vars import vars_update

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

    def get_responce(self, cluster: Cluster) -> tuple[str, Cluster, Cluster]:
        # initialize
        logs = cluster.batch_logs
        sample_log = cluster.sample_log
        cache_base = self.cache

        # Matching and Pruning
        new_cluster = Cluster()
        for log in cluster.logs:
            template, _, _ = cache_base.match_event(log)
            if template != "NoMatch":
                cluster, new_cluster = prune_from_cluster(template, cluster)
                if new_cluster.size >= 0 and new_cluster.size < cluster.size:
                    return template, cluster, new_cluster
                elif new_cluster.size == cluster.size:
                    cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
                    new_cluster = Cluster()

        # historical variables
        variable_cluster = Cluster()
        variable_cluster.logs = cache_base.variable_candidates
        if variable_cluster.logs != []:
            variable_cluster.varaible_sampling(5)
        variables = variable_cluster.batch_logs

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
        try:
            answer = self.chat(messages)
        except APITimeoutError:
            logger.debug("request timed out")
            answer = sample_log
        logger.debug(f"{answer=}")

        template = post_process(answer)

        if not verify_template(template):
            template = correct_single_template(sample_log)

        cluster, new_cluster = prune_from_cluster(template, cluster)
        if new_cluster.size == cluster.size:
            cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
            new_cluster = Cluster()
            template = correct_single_template(sample_log)

        return template, cluster, new_cluster

    def parse(
        self,
        logs: Sequence[str],
        /,
        batch_size: int = 10,
        chunk_size: int = 10_000,
        clustering_method: str = "dbscan",
    ) -> list[str]:
        log_chunk = []
        log_chunk_index = []
        caching = self.cache

        outputs: list[str | None] = [None for _ in range(len(logs))]

        # Parsing
        for batch in batched(enumerate(logs), n=chunk_size):
            for i, log in batch:
                result, template_id, _ = caching.match_event(log)
                if result != "NoMatch" and template_id != "NoMatch":
                    outputs[i] = caching.template_list[template_id]
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
                template, old_cluster, new_cluster = self.get_responce(old_cluster)
                # update clusters
                cluster_nums += process_new_cluster(new_cluster, clusters, batch_size)
                refer_log = old_cluster.logs[0]
                id: int
                if template not in caching.template_list:
                    if verify_template(template):
                        id, _, _ = caching.add_templates(
                            event_template=template,
                            insert=False,
                            refer_log=refer_log,
                        )
                        caching.variable_candidates.extend(
                            vars_update(
                                refer_log, template, caching.variable_candidates
                            )
                        )
                    else:
                        id, _, _ = caching.add_templates(
                            event_template=refer_log,
                            insert=False,
                            refer_log=refer_log,
                        )
                else:
                    id = caching.template_list.index(template)
                for i in old_cluster.indexs:
                    outputs[i] = caching.template_list[id]

        return [check_type(s, str) for s in outputs]


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

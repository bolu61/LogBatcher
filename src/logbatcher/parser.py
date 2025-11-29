from collections.abc import Iterable, Sequence
import json
import os
import time
from collections import Counter

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

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
from logbatcher.util import count_message_tokens, verify_template
from logbatcher.vars import vars_update


class Parser:
    def __init__(self, model):
        self.model = model
        self.token_list = [0, 0]
        self.time_consumption_llm = 0
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.cache = ParsingCache()

    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(10))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
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
        try:
            t0 = time.time()
            answer = self.chat(messages)
            print(messages)
            print(answer)
            self.token_list[0] += 1
            self.token_list[1] += count_message_tokens(messages, "gpt-4o-mini")
            self.time_consumption_llm += time.time() - t0
        except Exception as e:
            print("invoke LLM error", e)
            answer = sample_log

        template = post_process(answer)
        if not verify_template(template):
            template = correct_single_template(sample_log)

        cluster, new_cluster = prune_from_cluster(template, cluster)
        if new_cluster.size == cluster.size:
            cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
            new_cluster = Cluster()
            template = correct_single_template(sample_log)
        return template, cluster, new_cluster

    def __call__(
        self,
        batch: Sequence[str],
        batch_size=10,
        clustering_method="dbscan",
    ):
        logs = batch
        log_chunk = []
        log_chunk_index = []
        caching = self.cache

        # TODO: clarify types
        outputs_index: list[int] = [0 for _ in range(len(logs))]

        # Parsing
        for index, log in enumerate(logs):
            match_results = caching.match_event(log)
            if match_results[0] != "NoMatch":
                outputs_index[index] = match_results[1]
            else:
                log_chunk.append(log)
                log_chunk_index.append(index)

        # parsing start
        if clustering_method == "dbscan":
            # tokenize -> vectorize -> cluster -> reassign_clusters
            tokenized_logs = [tokenize(log) for log in log_chunk]
            labels, cluster_nums = cluster(vectorize(tokenized_logs))
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
        for label, log, index in zip(labels, log_chunk, log_chunk_index):
            clusters[label].append_log(log, index)

        # sorting
        clusters.sort(key=lambda cluster: len(cluster.logs), reverse=True)

        # batching
        [cluster.batching(batch_size) for cluster in clusters]

        # parsing
        for index, old_cluster in enumerate(clusters):
            template, old_cluster, new_cluster = self.get_responce(old_cluster)
            # update clusters
            cluster_nums += process_new_cluster(new_cluster, clusters, batch_size)
            refer_log = old_cluster.logs[0]
            if template not in caching.template_list:
                if verify_template(template):
                    id, _, _ = caching.add_templates(
                        event_template=template,
                        insert=False,
                        refer_log=refer_log,
                    )
                    caching.variable_candidates.extend(
                        vars_update(refer_log, template, caching.variable_candidates)
                    )
                else:
                    id, _, _ = caching.add_templates(
                        event_template=refer_log,
                        insert=False,
                        refer_log=refer_log,
                    )
            else:
                id = caching.template_list.index(template)
            for index in old_cluster.indexs:
                outputs_index[index] = id

        outputs: list[str] = [caching.template_list[i] for i in outputs_index]

        # Result
        t2 = time.time()
        print(f"parsing time: {t2 - t1}")
        print(f"idetified templates: {len(set(outputs))}")

        # output logs
        output_log_file = output_dir + f"{dataset}_full.log_structured.csv"
        df = pd.DataFrame({"Content": logs, "EventTemplate": outputs})
        df.to_csv(output_log_file, index=False)

        # output templates
        counter = Counter(outputs)
        items = list(counter.items())
        items.sort(key=lambda x: x[1], reverse=True)
        output_template_file = output_dir + f"{dataset}_full.template_structured.csv"
        template_df = pd.DataFrame(items, columns=["EventTemplate", "Occurrence"])
        template_df["EventID"] = [f"E{i + 1}" for i in range(len(template_df))]
        template_df[["EventID", "EventTemplate", "Occurrence"]].to_csv(
            output_template_file, index=False
        )

        # Save time cost
        time_cost_file = output_dir + "time_cost.json"
        time_table = {}
        if os.path.exists(time_cost_file):
            with open(time_cost_file, "r") as file:
                time_table = json.load(file)
        time_table[dataset] = {
            "InvocatingTime": parser.time_consumption_llm.__round__(3),
            "ParsingTime": (t2 - t1).__round__(3),
            "HitNum": caching.hit_num,
            "len_of_hashing_table": len(caching.hashing_cache),
            "TokenCount": parser.token_list,
        }
        with open(time_cost_file, "w") as file:
            json.dump(time_table, file)

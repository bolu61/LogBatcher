import re

from logbatcher.cache import Template
from logbatcher.cluster import Cluster


# TODO: refactor
def extract_variables(log: str, template: Template):
    log = re.sub(r"\s+", " ", log.strip())  # DS
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"
    matches = re.search(regex, log)
    if matches:
        return matches.groups()
    else:
        return None




# TODO: refactor
def prune_from_cluster(template, cluster):
    new_cluster = Cluster()
    logs, indexs = cluster.logs, cluster.indexs
    for log, index in zip(logs, indexs):
        if extract_variables(log, template) == None:
            new_cluster.append_log(log, index)
    if new_cluster.size != 0:
        old_logs = [log for log in logs if log not in new_cluster.logs]
        old_indexs = [index for index in indexs if index not in new_cluster.indexs]
        cluster.logs = old_logs
        cluster.indexs = old_indexs
        # print(f"prune {new_cluster.size} logs from {len(logs)} logs in mathcing process")
    return cluster, new_cluster


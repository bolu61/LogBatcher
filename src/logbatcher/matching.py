import re

from logbatcher.cache import Template
from logbatcher.cluster import Cluster


# TODO: refactor
def prune_from_cluster(template: Template, cluster: Cluster):
    new_cluster = Cluster()
    logs, indexs = cluster.logs, cluster.indexs
    for log, index in zip(logs, indexs):
        if extract_variables(log, template) is None:
            new_cluster.append_log(log, index)
    if new_cluster.size != 0:
        old_logs = [log for log in logs if log not in new_cluster.logs]
        old_indexs = [index for index in indexs if index not in new_cluster.indexs]
        cluster.logs = old_logs
        cluster.indexs = old_indexs
    return cluster, new_cluster


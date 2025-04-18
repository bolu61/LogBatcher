import json
import os
import re
import time
import pandas as pd
from collections import Counter, OrderedDict
from tqdm import tqdm
from logbatcher.vars import vars_update
from logbatcher.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters, process_new_cluster
from logbatcher.additional_cluster import hierichical_clustering,meanshift_clustering
from logbatcher.matching import matches_template, extract_variables
from logbatcher.util import verify_template

from logbatcher.parsing_cache import ParsingCache, tree_match

def single_dataset_paring(dataset, contents, output_dir, parser, batch_size = 10, chunk_size = 10000 , sample_method = 'dpp', clustering_method = 'dbscan', debug=True, min_size= 3, benchmark_mode = 0):

    clustering_method = 'hierarchical' if benchmark_mode == 1 else clustering_method # w/ hierarchical clustering
    clustering_method = 'meanshift' if benchmark_mode == 2 else clustering_method # w/ meanshift clustering
    
    sample_method = 'random' if benchmark_mode == 3 else sample_method # w/ random sampling
    sample_method = 'similar' if benchmark_mode == 4 else sample_method # w/ similar sampling

    chunk_size = 1000 if benchmark_mode == 5 else chunk_size # w/ 1000 chunk size
    chunk_size = 2000 if benchmark_mode == 5 else chunk_size # w/ 2000 chunk size
    chunk_size = 5000 if benchmark_mode == 6 else chunk_size # w/ 5000 chunk size
    chunk_size = 20000 if benchmark_mode == 7 else chunk_size # w/ 20000 chunk size
    
    chunk_size = 1 if benchmark_mode == 8 else chunk_size # w/ partitioning
    batch_size = 1 if benchmark_mode == 9 else batch_size # w/ batching

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logs = contents
    cache_pairs = {}
    log_chunk = []
    log_chunk_index = []
    caching = ParsingCache()
    print(f'Parsing {len(logs)} logs in dataset {dataset}...')

    outputs = [None for _ in range(len(logs))]
    outputs_index = [None for _ in range(len(logs))]
    
    # Parsing
    t1 = time.time()
    iterable = tqdm(enumerate(logs), total=len(logs), unit="log")
    for index, log in iterable:

        match_results = caching.match_event(log)
        if match_results[0] != "NoMatch":
            # outputs[index] = match_results[0]
            outputs_index[index] = match_results[1]
        else:
            log_chunk.append(log)
            log_chunk_index.append(index)
        

        # Parsing with LLM
        if len(log_chunk) == chunk_size or (len(log_chunk)!=0 and index == len(logs) - 1):
            # parsing start
            print(f'Parsing {len(log_chunk)} logs...') if debug else None
            if clustering_method == 'dbscan':
                # tokenize -> vectorize -> cluster -> reassign_clusters
                tokenized_logs = [tokenize(log) for log in log_chunk]
                labels, cluster_nums = cluster(vectorize(tokenized_logs))
                labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)
            elif clustering_method == 'hierarchical':
                labels, cluster_nums = hierichical_clustering(log_chunk)
            elif clustering_method == 'meanshift':
                labels, cluster_nums = meanshift_clustering(log_chunk)
            else:
                raise ValueError('Invalid clustering method')

            # create clusters
            clusters = [None for _ in range(cluster_nums)]
            for index, label in enumerate(labels):
                if clusters[label] is None:
                    clusters[label] = Cluster()
                clusters[label].append_log(log_chunk[index], log_chunk_index[index])

            # sorting
            clusters = sorted(clusters, key=lambda cluster: len(cluster.logs), reverse=True)

            # batching
            [cluster.batching(batch_size, sample_method, min_size) for cluster in clusters]

            # parsing
            # print(len(clusters), 'clusters identified') if debug else None  
            for index, old_cluster in enumerate(clusters):
                template, old_cluster, new_cluster = parser.get_responce(old_cluster, cache_base = caching)
                # update clusters
                cluster_nums += process_new_cluster(new_cluster, clusters, batch_size, sample_method, min_size)
                refer_log = old_cluster.logs[0]
                if template not in caching.template_list:
                    if verify_template(template):
                        if debug:
                            print('=' * 20)
                            print(f'New cluster processed, {len(set(caching.template_list))} templates identified till now:')
                            print(f'Refer Log: {refer_log}')
                            print(f'Output Template: {template}')
                        id, _, _ = caching.add_templates(event_template=template, insert=False, refer_log = refer_log)
                        caching.variable_candidates.extend(vars_update(refer_log, template, caching.variable_candidates))
                    else:
                        id, _, _ = caching.add_templates(event_template=refer_log, insert=False, refer_log = refer_log)
                else:
                    id = caching.template_list.index(template)
                for index in old_cluster.indexs:
                    outputs_index[index] = id
            log_chunk = []
            log_chunk_index = []
    
    print(caching.variable_candidates)
    outputs = [caching.template_list[i] for i in outputs_index]
    # Result
    t2 = time.time()
    print(f'parsing time: {t2 - t1}')
    print(f'idetified templates: {len(set(outputs))}')

    # output logs
    output_log_file = output_dir + f'{dataset}_full.log_structured.csv'
    df = pd.DataFrame({'Content': logs, 'EventTemplate': outputs})
    df.to_csv(output_log_file, index=False)

    # output templates
    counter = Counter(outputs)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    output_template_file = output_dir + f'{dataset}_full.template_structured.csv'
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(output_template_file, index=False)

    # Save time cost
    time_cost_file = output_dir + 'time_cost.json'
    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
    time_table[dataset] = {
        'InvocatingTime': parser.time_consumption_llm.__round__(3),
        'ParsingTime': (t2 - t1).__round__(3),
        'HitNum': caching.hit_num,
        'len_of_hashing_table': len(caching.hashing_cache)
    }
    with open(time_cost_file, 'w') as file:
        json.dump(time_table, file)
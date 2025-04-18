import json
import re
import string
import time
from openai import OpenAI
from together import Together
from tenacity import retry, stop_after_attempt, wait_random_exponential
from logbatcher.cluster import Cluster
from logbatcher.postprocess import post_process
from logbatcher.sample import nearest_k_pairs_from_log
from logbatcher.matching import extract_variables, matches_template, prune_from_cluster
from logbatcher.postprocess import correct_single_template
from logbatcher.util import verify_template, not_varibility
import httpx
import pdb

class Parser:

    def __init__(self, model, theme, config):

        self.model = model
        self.theme = theme
        self.time_consumption_llm = 0
        if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
            raise ValueError("Please provide your OpenAI API key and Together API key in the config.json file.")
        if 'gpt' in self.model:
            self.api_key = config['api_key_from_openai']
            self.client = OpenAI(
                api_key=self.api_key,
                base_url = 'https://api.996444.cn/v1'
            )
        else:
            self.api_key = config['api_key_from_together']
            self.client = Together(
                api_key=self.api_key   # api_key
            )

    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(10))
    def chat(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip('\n')

    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(10))
    def inference(self, prompt):
        retry_times = 0
        output = ''
        while True:
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.0,
                )
                output = response.choices[0].text.strip('\n')
            except Exception as e:
                print(e)
                retry_times += 1
                if retry_times > 3:
                    return output
            else:
                return output

    def get_responce(self, cluster, cache_base, sample_pairs=[], shot=0):

        # initialize
        logs = cluster.batch_logs
        sample_log = cluster.sample_log
        
        # Matching and Pruning
        new_cluster = Cluster()
        for log in cluster.logs:
            template, _, _ = cache_base.match_event(log)
            if template != "NoMatch":
                cluster, new_cluster = prune_from_cluster(
                    template, cluster)
                if new_cluster.size >= 0 and new_cluster.size < cluster.size:
                    return template, cluster, new_cluster
                elif new_cluster.size == cluster.size:
                    cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
                    new_cluster = Cluster()

        # historical variables
        variable_cluster = Cluster()
        variable_cluster.logs = cache_base.variable_candidates
        if variable_cluster.logs != []:
            variable_cluster.varaible_sampling(5, 'dpp')
        variables = variable_cluster.batch_logs
        # prompt format: instruction + (demonstration) + query(logs)
        instruction_1 = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. There might be no variables in the log message.\nPrint the input log's template delimited by backticks."

        variable_prompt = f' Historical variables: {variables}.' if variables != [] else ''

        # instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name', 'system_specific variables']." + variable_prompt + " There might be no variables in the log message.\nPrint the input log's template delimited by backticks."

        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with `{{placeholders}}` to extract the corresponding template. The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name']." + variable_prompt + " Constant text and strings should not be recognized as variables.\nPrint the input log's template delimited by backticks."

        if all(model_tpye not in self.model for model_tpye in ['gpt', 'instruct', 'chat']):
            query = 'Log message:\n' + \
                '\n'.join([f'`{log}`'for log in logs]) + '\nLog template: '
        else:
            # query = '\n'.join(logs)
            query = '' 
            for index, log in enumerate(logs):
                query += f'Log[{index+1}]: `{log}`\n'
            query.rstrip('\n')

        # invoke LLM
        if any(model_tpye in self.model for model_tpye in ['gpt', 'instruct', 'chat']):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content":  query}
            ]
            
            try:
                t0 = time.time()
                answer = self.chat(messages)
                print(sample_log)
                print(answer)
                self.time_consumption_llm += (time.time() - t0)
            except Exception as e:
                print("invoke LLM error", e)
                answer = sample_log
        else:
            prompt = f"{instruction}\n{query}"
            answer = self.inference(prompt)
        
        template = post_process(answer)
        if not verify_template(template):
            template = correct_single_template(sample_log)
        
        cluster, new_cluster = prune_from_cluster(template, cluster)
        if new_cluster.size == cluster.size:
            cluster.logs, cluster.indexs = new_cluster.logs, new_cluster.indexs
            new_cluster = Cluster()
            template = correct_single_template(sample_log)
        return template, cluster, new_cluster

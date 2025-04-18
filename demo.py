# python demo.py --model deepseek-chat --dataset Thunderbird --cache UnLeash --folder test_cache_UnLeash

import argparse
import json
from logbatcher.parsing_base import single_dataset_paring
from logbatcher.parser import Parser
from logbatcher.util import data_loader

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='the Large Lauguage model used in LogBatcher, default to be gpt-4o-mini.')
    parser.add_argument('--dataset', type=str, default='Proxifier')
    parser.add_argument('--folder', type=str, default='test')
    parser.add_argument('--benchmark_mode', type=int, default=0,
                        help='different setting')
    args = parser.parse_args()
    return args

# load api key, dataset format and parser
if __name__ == '__main__':
    args = set_args()
    model, dataset, folder_name =args.model,args.dataset, args.folder
    config = json.load(open('config.json', 'r'))
    if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
        print("Please provide your OpenAI API key and Together API key in the config.json file.")
        exit(0)

    parser = Parser(model, folder_name, config)

    # load contents from raw log file, structured log file or content list
    print(f"Loading {dataset} dataset")
    contents = data_loader(
        file_name=f"datasets/loghub-2.0/{dataset}/{dataset}_full.log",
        dataset_format= config['datasets_format'][dataset],
        file_format ='raw'
    )

    # parse logs
    single_dataset_paring(
        dataset=dataset,
        contents=contents,
        output_dir= f'outputs/parser/{folder_name}/',
        parser=parser,
        debug=True,
        benchmark_mode=args.benchmark_mode
    )
import os
import json
import argparse
from tabulate import tabulate

from utils.tasks import ALL_TASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tablefmt", type=str, default=None)
    parser.add_argument("--output", type=str, default="./output", help="The directory of the mteb evaluation output files")
    args = parser.parse_args()

    output_dir = args.output

    total_metrics, split_metrics  = [], {}

    headers = ["Categories",]

    table_data = [float('nan')]

    # rewrite here 2 control
    for task_type, tasks in ALL_TASKS:
        
        category_metrics = []
        cqadupstack_list = []

        for task_name in tasks:
            is_cqadupstack = False
            output_file = os.path.join(output_dir, task_type, f'{task_name}.json')
            with open(output_file, 'r') as file:
                json_obj = json.load(file)

            if task_type == "STS":                
                if task_name == "STS17":
                    result = json_obj['test']['en-en']['cos_sim']['spearman']     
                elif task_name == "STS22":
                    result = json_obj['test']['en']['cos_sim']['spearman']
                else:
                    result = json_obj['test']['cos_sim']['spearman']
            elif task_type == "Classification":
                result = json_obj["test"]["en"]["accuracy"] if "en" in json_obj["test"] else json_obj["test"]["accuracy"]
            elif task_type == "Clustering":
                result = json_obj["test"]["v_measure"]
            elif task_type == "PairClassification":
                result = json_obj["test"]["cos_sim"]["ap"]
            elif task_type == "Reranking":
                result = json_obj["test"]["map"]
            elif task_type == "Summarization":
                result = json_obj["test"]["cos_sim"]["spearman"]
            elif task_type == "Retrieval":
                result = json_obj["test"]["ndcg_at_10"]
                if task_name.startswith("CQADupstack"):
                    is_cqadupstack = True
            else:
                raise NotImplementedError
            
            if not is_cqadupstack:
                category_metrics.append(result)
                total_metrics.append(result)
            else:
                cqadupstack_list.append(result)
        
        if len(cqadupstack_list) > 0:
            result_cqadupstack = sum(cqadupstack_list) / len(cqadupstack_list)
            category_metrics.append(result_cqadupstack)
            total_metrics.append(result_cqadupstack)

        split_metrics[task_type] = category_metrics.copy()

        headers.append(f"{task_type}")

        table_data.append(round(sum(category_metrics) / len(category_metrics) * 100, 2))

    
    headers.append(f"Avg")
    table_data.append(round((sum(total_metrics)) / len(total_metrics) * 100, 2))


    if args.tablefmt is not None:
        # pretty pprint
        print(tabulate([table_data], headers, tablefmt=args.tablefmt))
    else:
        print(" | ".join([f"{_:.2f}" if _ is not float('nan') else " " for _ in table_data]))

# python run_eval.py --output_dir output








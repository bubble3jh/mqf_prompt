import wandb
from prettytable import PrettyTable
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="all")
parser.add_argument("--metric", type=str, default="spdp")
parser.add_argument("--shot", type=int, default=5)

args = parser.parse_args()

# wandb 로그인
wandb.login()

# 프로젝트 설정
if args.method == 'ours':
    project_name = "l2p_bp/frequency_prompt_tuning"
else:
    project_name = "l2p_bp/fewshot_transfer_baseline"

# target과 transfer 설정
settings = [
    ("bcg", "ppgbp"),
    ("bcg", "sensors"),
    ("ppgbp", "bcg"),
    ("ppgbp", "sensors"),
    ("sensors", "bcg"),
    ("sensors", "ppgbp")
]

sort_val=args.metric
print(f'sorted values by {sort_val}')

# spdp가 가장 작은 run을 가져오는 함수
def get_smallest_spdp_run(project, transfer, target, seed=None, baseline=None):
    api = wandb.Api()
    filters = {
        "config.transfer": transfer,
        "config.target": target,
    }
    if baseline == 'soso':
        filters = {
        "config.transfer": target,
        "config.target": target,
        }

    # 필터링 조건을 설정
    if seed is not None:
        filters['config.seed']=seed
        
    if baseline is not None:
        if baseline == 'head_off':
            filters['config.train_head']='false'
        else:
            filters['config.baseline']=baseline
        if baseline == 'soso':
            filters['config.baseline']='zero'
    if baseline is None:
        filters['config.add_prompts']='None'
        filters['config.train_head']='true'

    try:
        # 필터링된 runs를 가져옴
        runs = api.runs(project, filters=filters)
        if not runs:
            return None

        smallest_spdp_run = min(runs, key=lambda run: run.summary.get(f"{sort_val}", float('inf')))
        return smallest_spdp_run

    except Exception as e:
        print(f"Error fetching runs for target={target}, transfer={transfer}: {e}")
        return None

def calculate_stats(values):
    mean = round(np.mean(values), 3)
    std = round(np.std(values), 3)
    return mean, std

if args.method == 'ours':
    # 결과를 담을 표 설정
    table = PrettyTable()
    table.field_names = ["Transfer", "Target", "Avg SPDP (mean ± std)", "Avg GAL (mean ± std)"]

    # 각 설정에 대해 spdp가 가장 작은 run을 가져와서 표에 추가
    seeds = [0]

    for target, transfer in settings:
        spdp_values = []
        gal_values = []
        
        for seed in seeds:
            run = get_smallest_spdp_run(project_name, transfer, target, seed)
            if run:
                spdp = run.summary.get("spdp", "N/A")
                gal = run.summary.get("gal", "N/A")
                if spdp != "N/A" and gal != "N/A":
                    spdp_values.append(spdp)
                    gal_values.append(gal)
            else:
                print("Look seed : ", seed)
        
        if spdp_values and gal_values:
            avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
            avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
            table.add_row([transfer, target,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
        else:
            table.add_row([transfer, target, "N/A", "N/A"])

    print(table)
elif args.method == 'all':

    if args.shot == 5:
        project_names = [f"l2p_bp/fewshot_transfer_baseline", f"l2p_bp/freq_prompt_sym_shot{args.shot}", f"l2p_bp/freq_prompt_sym_shot{args.shot}"]
    else:
        project_names = [f"l2p_bp/fewshot_transfer_10shot_baseline", f"l2p_bp/freq_prompt_sym_shot{args.shot}", f"l2p_bp/freq_prompt_sym_shot{args.shot}"]

    method_names = ['baseline', 'head_off', 'ours']
    
    table = PrettyTable()
    table.field_names = ["Transfer", "Target", 'Method', "Avg SPDP (mean ± std)", "Avg GAL (mean ± std)"]

    seeds = [0]

    best_hy_commands = []
    for target, transfer in settings:
        for i, project_name in enumerate(project_names):
            method_name = method_names[i]
            if i==0:
                for baseline in ['soso', 'zero', 'ft', 'scratch', 'lp']:
                    spdp_values = []
                    gal_values = []
                    method_name = baseline
                    print(f"Start: {transfer} => {target} : {method_name}")
                    for seed in seeds:
                        run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline)
                        if run:
                            spdp = run.summary.get("spdp", "N/A")
                            gal = run.summary.get("gal", "N/A")
                            if spdp != "N/A" and gal != "N/A":
                                spdp_values.append(spdp)
                                gal_values.append(gal)
                                best_hy_commands.append(f"{run.metadata['executable']} {run.metadata['program']} {' '.join(run.metadata['args'])}")
                        else:
                            print("Look seed : ", seed)
                    
                    if spdp_values and gal_values:
                        avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                        avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                        table.add_row([transfer, target, method_name,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
                    else:
                        table.add_row([transfer, target, method_name, "N/A", "N/A"])
                    print(f"Done: {transfer} => {target} : {method_name}")
            elif i == 1: # head off
                spdp_values = []
                gal_values = []
                print(f"Start: {transfer} => {target} : {method_name}")
                for seed in seeds:
                    run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline=method_name)
                    if run:
                        spdp = run.summary.get("spdp", "N/A")
                        gal = run.summary.get("gal", "N/A")
                        if spdp != "N/A" and gal != "N/A":
                            spdp_values.append(spdp)
                            gal_values.append(gal)
                            best_hy_commands.append(f"{run.metadata['executable']} {run.metadata['program']} {' '.join(run.metadata['args'])}")
                    else:
                        print("Look seed : ", seed)
                    
                if spdp_values and gal_values:
                    avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                    avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                    table.add_row([transfer, target, method_name,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
                else:
                    table.add_row([transfer, target, method_name, "N/A", "N/A"])
                print(f"Done: {transfer} => {target} : {method_name}")
            else: # ours
                print(f"Start: {transfer} => {target} : {method_name}")
                spdp_values = []
                gal_values = []
                for seed in seeds:
                    run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline=None)
                    if run:
                        spdp = run.summary.get("spdp", "N/A")
                        gal = run.summary.get("gal", "N/A")
                        if spdp != "N/A" and gal != "N/A":
                            spdp_values.append(spdp)
                            gal_values.append(gal)
                            best_hy_commands.append(f"{run.metadata['executable']} {run.metadata['program']} {' '.join(run.metadata['args'])}")
                    else:
                        print("Look seed : ", seed)
                    
                if spdp_values and gal_values:
                    avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                    avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                    table.add_row([transfer, target, method_name,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
                else:
                    table.add_row([transfer, target, method_name, "N/A", "N/A"])
                print(f"Done: {transfer} => {target} : {method_name}")
    
    with open(f"results_table_shot{args.shot}.txt", "w") as text_file:
        text_file.write(str(table))

    with open(f"./best_hy_command_shot{args.shot}.sh", "w") as f:
        for c in best_hy_commands:
            f.write(c + "\n")

elif args.method == 'repeat':

    project_names = ["l2p_bp/fewshot_transfer_baseline", "exp_repeat_ours_best_spdp"]
    method_names = ['baseline', 'ours']
    
    table = PrettyTable()
    table.field_names = ["Transfer", "Target", 'Method', "Avg SPDP (mean ± std)", "Avg GAL (mean ± std)"]

    seeds = [0,1,2]

    for target, transfer in settings:
        for i, project_name in enumerate(project_names):
            method_name = method_names[i]
            if i==0:
                for baseline in ['ft', 'scratch', 'lp']:
                    spdp_values = []
                    gal_values = []
                    method_name = baseline
                    for seed in seeds:
                        run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline)
                        if run:
                            spdp = run.summary.get("spdp", "N/A")
                            gal = run.summary.get("gal", "N/A")
                            if spdp != "N/A" and gal != "N/A":
                                spdp_values.append(spdp)
                                gal_values.append(gal)
                        else:
                            print("Look seed : ", seed)
                    
                    if spdp_values and gal_values:
                        avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                        avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                        table.add_row([transfer, target, method_name,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
                    else:
                        table.add_row([transfer, target, method_name, "N/A", "N/A"])
            else:
                spdp_values = []
                gal_values = []
                for seed in seeds:
                    run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline=None)
                    if run:
                        spdp = run.summary.get("spdp", "N/A")
                        gal = run.summary.get("gal", "N/A")
                        if spdp != "N/A" and gal != "N/A":
                            spdp_values.append(spdp)
                            gal_values.append(gal)
                    else:
                        print("Look seed : ", seed)
                    
                if spdp_values and gal_values:
                    avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                    avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                    table.add_row([transfer, target, method_name,  f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
                else:
                    table.add_row([transfer, target, method_name, "N/A", "N/A"])
    
    with open("results_table.txt", "w") as text_file:
        text_file.write(str(table))

else:
    table = PrettyTable()
    table.field_names = ["Transfer", "Target", "Baseline", "Avg SPDP (mean ± std)", "Avg GAL (mean ± std)"]

    # 각 설정에 대해 spdp가 가장 작은 run을 가져와서 표에 추가
    baselines = ['ft', 'scratch', 'lp']
    seeds = [0]

    for target, transfer in settings:
        for baseline in baselines:
            spdp_values = []
            gal_values = []
            
            for seed in seeds:
                run = get_smallest_spdp_run(project_name, transfer, target, seed, baseline)

                if run:
                    spdp = run.summary.get("spdp", "N/A")
                    gal = run.summary.get("gal", "N/A")
                    if spdp != "N/A" and gal != "N/A":
                        spdp_values.append(spdp)
                        gal_values.append(gal)
                else:
                    print("Look seed : ", seed)

            if spdp_values and gal_values:
                avg_spdp_mean, avg_spdp_std = calculate_stats(spdp_values)
                avg_gal_mean, avg_gal_std = calculate_stats(gal_values)
                table.add_row([transfer,target, baseline, f"{avg_spdp_mean} ± {avg_spdp_std}", f"{avg_gal_mean} ± {avg_gal_std}"])
            else:
                table.add_row([transfer, target, baseline, "N/A", "N/A"])

    print(table)

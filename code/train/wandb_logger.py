# %%
import wandb
import pandas as pd
import numpy as np

# wandb API key 설정
wandb.login()

# 프로젝트 및 필터 설정
project_name = "fewshot_transfer_gumbel"
filter = {
    "method": "original",
    "shots": 5
}

# 불러올 필드 설정
fields = ["seed", "baseline", "spdp", "gal", "transfer", "target"]

# baseline과 transfer, target 설정
baselines = ["ft", "lp", "scratch"]
transfers_targets = [
    {"transfer": "ppgbp", "target": "bcg"},
    {"transfer": "sensors", "target": "bcg"},
    {"transfer": "bcg", "target": "ppgbp"},
    {"transfer": "sensors", "target": "ppgbp"},
    {"transfer": "bcg", "target": "sensors"},
    {"transfer": "ppgbp", "target": "sensors"}
]

# 결과를 저장할 딕셔너리 초기화
results = {(baseline, tt["transfer"], tt["target"]): {"spdp": [], "gal": []} for baseline in baselines for tt in transfers_targets}

# 각 시드에 대해 wandb 데이터 가져오기
runs = wandb.Api().runs(f"{project_name}")

# 시드별, baseline별로 spdp가 가장 낮은 run 찾기
for seed in range(5):
    seed_runs = [run for run in runs if run.config.get("method") == filter["method"] and run.config.get("shots") == filter["shots"] and run.config.get("seed") == seed]
    for baseline in baselines:
        for tt in transfers_targets:
            baseline_tt_seed_runs = [run for run in seed_runs if run.config.get("baseline") == baseline and run.config.get("transfer") == tt["transfer"] and run.config.get("target") == tt["target"]]
            if baseline_tt_seed_runs:
                min_spdp_run = min(baseline_tt_seed_runs, key=lambda run: run.summary.get("spdp", float("inf")))
                min_spdp = min_spdp_run.summary.get("spdp")
                corresponding_gal = min_spdp_run.summary.get("gal")
                results[(baseline, tt["transfer"], tt["target"])]["spdp"].append(min_spdp)
                results[(baseline, tt["transfer"], tt["target"])]["gal"].append(corresponding_gal)
# %%

# 각 baseline, transfer, target 조합에 대해 평균 및 분산 계산
final_results = {}
for key, value in results.items():
    baseline, transfer, target = key
    spdp_list = value["spdp"]
    gal_list = value["gal"]
    
    spdp_mean = np.mean(spdp_list) if spdp_list else None
    spdp_variance = np.var(spdp_list) if spdp_list else None
    gal_mean = np.mean(gal_list) if gal_list else None
    gal_variance = np.var(gal_list) if gal_list else None
    
    final_results[key] = {
        "spdp": f"{spdp_mean:.2f}±{spdp_variance:.2f}" if spdp_mean is not None else "N/A",
        "gal": f"{gal_mean:.2f}±{gal_variance:.2f}" if gal_mean is not None else "N/A"
    }

# DataFrame으로 변환 및 재구성
final_df = pd.DataFrame(final_results).transpose()
final_df.index = pd.MultiIndex.from_tuples(final_df.index, names=["Baseline", "Transfer", "Target"])
final_df = final_df.unstack(level=[1, 2])

# 컬럼명 재구성
final_df.columns = final_df.columns.swaplevel(0, 2).swaplevel(0, 1)
final_df.columns = pd.MultiIndex.from_tuples([(t[1], t[2], t[0]) for t in final_df.columns], names=["Transfer", "Target", "Metric"])

final_df
# %%
csv_file_path = "./final_results.csv"
final_df.to_csv(csv_file_path)
import wandb
from prettytable import PrettyTable

# wandb 로그인
wandb.login()

# 프로젝트 설정
project_name = "l2p_bp/fewshot_transfer_real_head"

# target과 transfer 설정
settings = [
    ("bcg", "ppgbp"),
    ("bcg", "sensors"),
    ("ppgbp", "bcg"),
    ("ppgbp", "sensors"),
    ("sensors", "bcg"),
    ("sensors", "ppgbp")
]


sort_val='gal'
print(f'sorted values by {sort_val}')

# spdp가 가장 작은 run을 가져오는 함수
def get_smallest_spdp_run(project, target, transfer):
    api = wandb.Api()
    # 필터링 조건을 설정
    filters = {
        "config.target": target,
        "config.transfer": transfer
        # 필요시 다른 조건 추가
    }
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

# 결과를 담을 표 설정
table = PrettyTable()
table.field_names = ["Target", "Transfer", "SPDP", "GAL"]

# 각 설정에 대해 spdp가 가장 작은 run을 가져와서 표에 추가
for target, transfer in settings:
    run = get_smallest_spdp_run(project_name, target, transfer)
    if run:
        spdp = run.summary.get("spdp", "N/A")
        gal = run.summary.get("gal", "N/A")
        table.add_row([target, transfer, spdp, gal])
    else:
        table.add_row([target, transfer, "N/A", "N/A"])

# 표 출력
print(table)

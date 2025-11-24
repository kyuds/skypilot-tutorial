from dataclasses import dataclass, asdict

import sky
from sky.jobs.client import sdk as jobs_sdk


@dataclass
class HyperParamConfig:
    run_name: str
    batch_size: int
    learning_rate: float


configs = [
    HyperParamConfig("run-v1", 8, 2e-15),
    HyperParamConfig("run-v2", 16, 2e-15),
]

for config in configs:
    task = sky.Task.from_yaml("train.yaml")
    task.update_envs(asdict(config))
    jobs_sdk.launch(task, name=f"sky-task-{config.run_name}")
    print(f"Submitted hyperparameter tuning for ${config}")

import argparse
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class HyperParamConfig:
    run_name: str
    batch_size: int
    learning_rate: float

    def to_flag(self) -> str:
        return (
            f"--env RUN_NAME={self.run_name} "
            f"--env BATCH_SIZE={self.batch_size} "
            f"--env LEARNING_RATE={self.learning_rate}"
        )


def run(pool: str):
    configs = [
        HyperParamConfig("run-v1", 8, 2e-15),
        HyperParamConfig("run-v2", 16, 2e-15),
        HyperParamConfig("run-v3", 32, 2e-15),
    ]

    for config in configs:
        cmd = f"sky jobs launch --pool {pool} train-pool.yaml {config.to_flag()} -d"
        print(f"running: {cmd}")
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        type=str,
        required=True,
        help="SkyPilot pool name",
    )

    run(parser.parse_args().pool)

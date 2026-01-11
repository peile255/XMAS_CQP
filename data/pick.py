import pandas as pd
from pathlib import Path

# 🔥 关键：以当前脚本所在目录为基准
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR  # data 目录
OUTPUT_DIR = BASE_DIR / "rq1_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SAMPLE_SIZE = 200



def sample_commits(
    input_csv: Path,
    output_csv: Path,
    sample_size: int,
    random_seed: int = 42
):
    """
    Randomly sample commits with fixed seed for RQ1.
    """
    df = pd.read_csv(input_csv)

    if len(df) < sample_size:
        raise ValueError(
            f"{input_csv.name} has only {len(df)} rows, "
            f"less than requested sample_size={sample_size}"
        )

    sampled_df = df.sample(
        n=sample_size,
        random_state=random_seed
    ).reset_index(drop=True)

    sampled_df.to_csv(output_csv, index=False)

    print(
        f"[OK] {input_csv.name}: "
        f"sampled {sample_size} / {len(df)} commits → {output_csv}"
    )


if __name__ == "__main__":
    datasets = {
        "openstack": INPUT_DIR / "openstack.csv",
        "qt": INPUT_DIR / "qt.csv",
    }

    for name, path in datasets.items():
        sample_commits(
            input_csv=path,
            output_csv=OUTPUT_DIR / f"{name}_rq1_sample.csv",
            sample_size=SAMPLE_SIZE,
            random_seed=RANDOM_SEED
        )

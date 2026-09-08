import pandas as pd
import yaml
from evaluation.experiments import run_system_a_always_on, run_system_b_event_driven, run_system_c_eventvision

def run_comparative_benchmark(video_path: str = "data/videos/sample_surveillance.mp4", config_path: str = "configs/config.yaml") -> pd.DataFrame:
    print(f"Starting EventVision Comparative Benchmark on {video_path}...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    res_a = run_system_a_always_on(video_path)
    res_b = run_system_b_event_driven(video_path, config)
    res_c = run_system_c_eventvision(video_path, config)

    df = pd.DataFrame([res_a, res_b, res_c])
    return df

if __name__ == "__main__":
    df_results = run_comparative_benchmark()
    print("\n================== BENCHMARK RESULTS ==================")
    print(df_results.to_string(index=False))

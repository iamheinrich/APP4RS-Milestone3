import subprocess


def run_training():
    # Define the command to run your training script with arguments
    command = [
        "python", "experiments.py",  # replace with your script's filename
        "--task", "slc",
        "--logging_dir", "untracked-files/logging_dir",
        "--dataset", "tiny-BEN",
        "--lmdb_path", "./untracked-files/BigEarthNet/BigEarthNet.lmdb",
        "--metadata_parquet_path", "./untracked-files/BigEarthNet/BigEarthNet.parquet",
        "--num_channels", "12",
        "--num_classes", "19",
        "--num_workers", "4",
        "--batch_size", "32",
        "--arch_name", "CustomCNN",
        "--epochs", "30",
        "--learning_rate", "0.001",
        "--weight_decay", "0.01",

        "--max_lr", "0.001",  #added   # should be bigger than learning_rate
        "--pct_start", "0.3", #added  # 0.3 in benchmark task
        "--patience", "5" #added  # 0.3 in benchmark task
    ]

    # Run the command synchronously and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the process completed successfully
    if result.returncode == 0:
        print("Training completed successfully.")
        print(result.stdout)
        print("Error:")
        print(result.stdout)
    else:
        print("Training output:")
        print(result.stdout)
        print("Error occurred during training:")
        print(result.stderr)


if __name__ == "__main__":
    run_training()

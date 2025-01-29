import subprocess
import os

class ExperimentRunner:
    def __init__(self):
        self.base_command = ["python", "experiments.py"]
        self.datasets = {
            "tiny-BEN": {
                "num_channels": 12,
                "num_classes": 1, #TODO: Change classes
                "lmdb_path": "/untracked-files/BigEarthNet/BigEarthNet.lmdb",
                "metadata_path": "/untracked-files/BigEarthNet/BigEarthNet.parquet"
            },
            "EuroSAT": {
                "num_channels": 13,
                "num_classes": 1, #TODO: Change classes
                "lmdb_path": "/untracked-files/EuroSAT/EuroSAT.lmdb",
                "metadata_path": "/untracked-files/EuroSAT/EuroSAT.parquet"
            },
            "Caltech-101": {
                "num_channels": 3,
                "num_classes": 1, #TODO: Change classes
                "lmdb_path": "./untracked-files/caltech101", #TODO: Adapt path
                "metadata_path": None
            }
        }

    def _run_command(self, command: list[str]) -> None:
        """Run a command and handle corresponding output."""
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Training completed successfully.")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            if result.stderr:
                print("Error messages:")
                print(result.stderr)
        else:
            print("Training failed!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            if result.stderr:
                print("Error messages:")
                print(result.stderr)

    def run_augmentation_study(self) -> None:
        """Run augmentation study experiment."""
        augmentations = [
            "--apply_random_resize_crop",
            "--apply_cutout",
            "--apply_brightness",
            "--apply_contrast",
            "--apply_grayscale"
        ]
        
        for dataset_name, config in self.datasets.items():
            # Construct base command for this dataset
            base_cmd = self.base_command + [
                "--experiment_type", "augmentation_study", #TODO: Needs to be adapted
                "--dataset", dataset_name,
                "--num_channels", str(config["num_channels"]),
                "--num_classes", str(config["num_classes"]),
                "--lmdb_path", config["lmdb_path"],
                "--pretrained", "False"
            ]
            
            if config["metadata_path"]:
                base_cmd.extend(["--metadata_parquet_path", config["metadata_path"]])
            
            # Run without augmentations (baseline)
            print(f"\nRunning baseline experiment for {dataset_name}")
            self._run_command(base_cmd)
            
            # Run with each augmentation individually
            for aug in augmentations:
                print(f"\nRunning {aug.lstrip('--')} experiment for {dataset_name}")
                aug_cmd = base_cmd + [aug]
                self._run_command(aug_cmd)

def main():
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("untracked-files", exist_ok=True)
    
    # Run augmentation study experiment
    print("Starting Data Augmentation Study...")
    runner.run_augmentation_study()

if __name__ == "__main__":
    main()

import subprocess
import os

class ExperimentRunner:
    """ Class to run experiments """
    def __init__(self):
        self.base_command = ["python", "experiments.py"]
        self.datasets = {
            "tiny-BEN": {
                "task": "mlc",
                "num_channels": "12",
                "num_classes": "19",
                "lmdb_path": "./untracked-files/BigEarthNet/BigEarthNet.lmdb",
                "metadata_path": "./untracked-files/BigEarthNet/BigEarthNet.parquet"
            },
            "EuroSAT": {
                "task": "slc",
                "num_channels": "13", #TODO: Verify
                "num_classes": "10",
                "lmdb_path": "./untracked-files/EuroSAT/EuroSAT.lmdb",
                "metadata_path": "./untracked-files/EuroSAT/EuroSAT.parquet"
            },
            "Caltech-101": {
                "task": "slc",
                "num_channels": "3",
                "num_classes": "101",
                "lmdb_path": "./untracked-files/caltech101",
                "metadata_path": None
            }
        }

    def _run_command(self, command: list[str]) -> None:
        """Run a command and handle corresponding output."""
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True) # Subprocesses are executed synchronously
        
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

        fixed_parameters = [
            "--weight_decay", "0.01",
            "--batch_size", "32",
            "--num_workers", "4",
            "--epochs", "30",
            "--pct_start", "0.3",

            "--logging_dir", "untracked-files/logging_dir",
            "--patience","5",
            "--experiment_type", "augmentation_study",
            #"--early_stopping",

            "--arch_name","ResNet18",#"--pretrained","--dropout"
        ]

        
        for dataset_name, config in self.datasets.items():
            if dataset_name == "Caltech-101":
                lr = ["--learning_rate", "0.025","--max_lr", "0.125"]
            else:
                lr = ["--learning_rate", "0.001","--max_lr", "0.01",]

            # Construct base command for this dataset
            base_cmd = self.base_command + [
                "--task", config["task"],
                "--dataset", dataset_name,
                "--num_channels", config["num_channels"],
                "--num_classes", config["num_classes"],
                "--lmdb_path", config["lmdb_path"],
                #"--pretrained", "False",
            ] + fixed_parameters + lr
            
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
    
    def run_feature_extraction_study(self) -> None:
        """Task 7: Run feature extraction and t-SNE visualization."""
        fixed_parameters = [
            "--learning_rate", "0.001",
            "--weight_decay", "0.01",
            "--batch_size", "32",
            "--num_workers", "4",
            "--epochs", "30",
            "--pct_start", "0.3",

            "--logging_dir", "untracked-files/logging_dir",
            "--patience","5",
            "--max_lr", "0.01",
            "--experiment_type", "feature_extraction_study",
            #"--early_stopping",

            "--arch_name","ResNet18",#"--pretrained","--dropout"
            "--task", "slc",
            "--dataset", "EuroSAT",
            "--num_channels", "13",
            "--num_classes", "10",
            "--lmdb_path", "./untracked-files/EuroSAT/EuroSAT.lmdb",
            "--metadata_parquet_path", "./untracked-files/EuroSAT/EuroSAT.parquet"
        ]

        self._run_command(self.base_command + fixed_parameters)
    
    def run_multi_model_benchmark_experiment(self) -> None:
        """Task 6: Run multi-model benchmark experiment."""
        fixed_parameters = [
            "--learning_rate", "0.001",
            "--weight_decay", "0.01",
            "--batch_size", "32",
            "--num_workers", "4",
            "--epochs", "30",
            "--pct_start", "0.3",

            "--logging_dir", "untracked-files/logging_dir",
            "--apply_flip",
            "--patience","5",
            "--max_lr", "0.01",
            "--experiment_type", "multi_model_benchmark",
            "--early_stopping",
        ]

        arch_name_and_pretrained = [
            ["--arch_name","CustomCNN"],
            ["--arch_name","ResNet18"],
            ["--arch_name","ResNet18","--pretrained"],
            ["--arch_name","ConvNeXt-Nano"],
            ["--arch_name","ConvNeXt-Nano","--pretrained"],
            ["--arch_name","ViT-Tiny"],
            ["--arch_name","ViT-Tiny","--pretrained"],
        ]

        for dataset_name, ds_config in self.datasets.items():
            if(dataset_name=="Caltech-101"):
                continue
            base_cmd = self.base_command + fixed_parameters + [
                "--task", ds_config["task"],
                "--dataset", dataset_name,
                "--num_channels", ds_config["num_channels"],
                "--num_classes", ds_config["num_classes"],
                "--lmdb_path", ds_config["lmdb_path"],
                "--metadata_parquet_path", ds_config["metadata_path"]
            ]

            for arch_and_pre in arch_name_and_pretrained:

                transformer_sensitive_cmd = arch_and_pre + (["--apply_resize112"] if arch_and_pre[1]=="ViT-Tiny" else [])

                print(f"\nRunning {arch_and_pre[1]},{arch_and_pre[-1]},Dropout,{dataset_name}")
                self._run_command(base_cmd + transformer_sensitive_cmd)
                print(f"\nRunning {arch_and_pre[1]},{arch_and_pre[-1]},Dropless,{dataset_name}")
                self._run_command(base_cmd + transformer_sensitive_cmd + ["--dropout"])

def main():
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    #os.makedirs("untracked-files", exist_ok=True)
    
    # Run augmentation study experiment
    print("Task 6: Starting Multi Model Benchmark Experiment...")
    runner.run_multi_model_benchmark_experiment()

    # Run augmentation study experiment
    #print("Task 7: Starting Feature Extraction Study...")
    #runner.run_feature_extraction_study()

    # Run augmentation study experiment
    print("Task 9: Starting Data Augmentation Study...")
    runner.run_augmentation_study()


if __name__ == "__main__":
    main()

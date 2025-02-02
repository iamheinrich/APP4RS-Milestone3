import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data.data_EuroSAT import EUROSAT_CLASSES
import os

def run_tsne_analysis(features_dir, output_dir):
    """Run t-SNE analysis on extracted features and save visualizations with optimized resolution."""
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in [5, 10]:
        print(f"\nProcessing epoch {epoch}...")
        
        # Load data
        features = np.load(os.path.join(features_dir, f"epoch_{epoch}_features.npy"))
        labels = np.load(os.path.join(features_dir, f"epoch_{epoch}_labels.npy"))
        
        # Verify dimensions match (should always be true due to collection process)
        assert len(features) == len(labels), f"Unexpected mismatch between features ({len(features)}) and labels ({len(labels)})"
        
        # Perform t-SNE
        print("Running t-SNE...")
        tsne = TSNE(random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Reduce plot size: 6.4 x 4.8 inches = 480p, lower DPI to 100
        plt.figure(figsize=(6.4, 4.8))  # 640x480 pixels at DPI=100
        colors = plt.cm.rainbow(np.linspace(0, 1, len(EUROSAT_CLASSES)))
        
        for i, (label, color) in enumerate(zip(EUROSAT_CLASSES, colors)):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[color], label=label, alpha=0.6, s=8)  # Reduce point size
        
        plt.title(f't-SNE Visualization (Epoch {epoch})', fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        # Save with lower DPI
        plt.savefig(os.path.join(output_dir, f"tsne_epoch_{epoch}.jpg"), dpi=300, bbox_inches='tight')
        print(f"Saved plot for epoch {epoch} at 480p resolution")
        plt.close()

if __name__ == "__main__":
    run_tsne_analysis("./untracked-files/features/extracted", "./attachments/")#plots")
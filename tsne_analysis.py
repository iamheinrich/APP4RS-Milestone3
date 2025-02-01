import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data.data_EuroSAT import EUROSAT_CLASSES
import os

def run_tsne_analysis(features_dir, output_dir):
    """Run t-SNE analysis on extracted features and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in [5, 10]:
        print(f"\nProcessing epoch {epoch}...")
        
        # Load data
        features = np.load(os.path.join(features_dir, f"epoch_{epoch}_features.npy"))
        labels = np.load(os.path.join(features_dir, f"epoch_{epoch}_labels.npy"))
        
        # Perform t-SNE
        print("Running t-SNE...")
        tsne = TSNE(random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(EUROSAT_CLASSES)))
        
        for i, (label, color) in enumerate(zip(EUROSAT_CLASSES, colors)):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[color], label=label, alpha=0.6)
        
        plt.title(f't-SNE Visualization (Epoch {epoch})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(output_dir, f"tsne_epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot for epoch {epoch}")
        plt.close()

if __name__ == "__main__":
    run_tsne_analysis("./untracked-files/features/extracted", "./attachments/")#plots")

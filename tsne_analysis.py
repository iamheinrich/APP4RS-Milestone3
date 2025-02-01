import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data.data_EuroSAT import EUROSAT_CLASSES
import os

# Create output directory
os.makedirs("./features/plots", exist_ok=True)
os.makedirs("./features/extracted", exist_ok=True) 

# Process both epochs
for epoch in [5, 10]:
    print(f"\nProcessing epoch {epoch}...")
    
    # Load data
    features = np.load(f"./features/extracted/epoch_{epoch}_features.npy")
    labels = np.load(f"./features/extracted/epoch_{epoch}_labels.npy")
    
    # Perform t-SNE
    print("Running t-SNE...")
    tsne = TSNE(random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(EUROSAT_CLASSES)))
    
    for i, (label, color) in enumerate(zip(EUROSAT_CLASSES, colors)):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[color], label=label, alpha=0.6)
    
    plt.title(f't-SNE Visualization (Epoch {epoch})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save
    plt.savefig(f"./attachment/plots/tsne_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot for epoch {epoch}")
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class UniverseVisualizer:
    """
    A visualization suite designed to analyze the composition and 
    statistical distributions of a multimodal universe manifest.
    """

    def __init__(self, manifest: pd.DataFrame):
        """
        Initializes the visualizer with a dataset manifest.

        Args:
            manifest (pd.DataFrame): The output from MultimodalDatasetFactory.combine_modalities.
                Expected to have columns: 'subset', 'modality', and 'path'.
        """
        self.df = manifest
        # Set a clean, academic aesthetic for astronomical/physics data
        sns.set_theme(style="whitegrid", palette="magma")

    def plot_modality_balance(self):
        """
        Generates a categorical bar chart showing the volume of subsets 
        contributed by each modality.
        
        Logic:
            Provides a high-level view of dataset "bias" (e.g., if you have 
            90% images and only 10% time series).
        """
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(data=self.df, x='modality')
        ax.set_title("Distribution of Subsets per Modality", fontsize=14)
        plt.show()

    def plot_temporal_distribution(self, ts_metadata: List[Dict]):
        """
        Visualizes the distribution of sampling rates across time series subsets.

        Args:
            ts_metadata (List[Dict]): A list of dictionaries containing 
                'sampling_rate_hz' for the registered time series.
        
        Logic:
            Uses a Kernel Density Estimate (KDE) to show where your 
            temporal resolution is clustered. Crucial for identifying 
            if your model is over-exposed to high-frequency data.
        """
        if not ts_metadata:
            print("No time series metadata provided for visualization.")
            return

        rates = [m['sampling_rate_hz'] for m in ts_metadata if 'sampling_rate_hz' in m]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(rates, kde=True, color="skyblue")
        plt.axvline(np.mean(rates), color='red', linestyle='--', label=f'Mean: {np.mean(rates):.2f}Hz')
        plt.title("Time Series Sampling Rate Distribution")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        plt.show()

    def plot_modality_intersection(self):
        """
        Creates a 'Matrix' or 'UpSet' style plot to show how many subsets 
        overlap across different modalities.
        
        Note:
            In the Multimodal Universe, you often want to know if you have 
            both an image AND a time series for the same celestial object.
        """
        # Simplistic implementation using a heatmap of counts
        cross_tab = pd.crosstab(self.df['subset'], self.df['modality'])
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(cross_tab.T, cmap="YlGnBu", cbar=False, linewidths=.5)
        plt.title("Subset-Modality Availability Matrix")
        plt.show()
        
        
    def plot_sample_images(self, examples: List[Dict[str, Any]], n: int = 4):
        """
        Renders a horizontal montage of RGB astronomical postage stamps.
        
        Use Case: 
            Visual inspection of morphological features (spirals, ellipticals, 
            mergers) and checking for image artifacts or saturation.
        
        Expertise Note: 
            In Multimodal Universe (MU) datasets, images are often stored in 
            Channels-First (C, H, W) format. This method ensures the correct 
            transposition to (H, W, C) for standard matplotlib rendering.
        
        Args:
            examples (List[Dict]): A list of data dictionaries containing 'rgb' 
                tensors and 'object_id' strings.
            n (int): Number of images to display (default: 4).
        """
        plt.figure(figsize=(15, 5))
        for i, item in enumerate(examples[:n]):
            plt.subplot(1, n, i+1)
            # Transpose if data is (C, H, W)
            img = item['rgb']
            if img.shape[0] < img.shape[2]: # Simple check for C, H, W
                img = np.transpose(img, (1, 2, 0))
                
            plt.imshow(img)
            plt.title(f"ID: {item.get('object_id', 'Unknown')}", fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_photometric_sed(self, example: Dict[str, Any]):
        """
        Plots the Spectral Energy Distribution (SED) using multi-band photometry.
        
        Use Case: 
            Essential for identifying galaxy types (Star-forming vs. Quiescent) 
            and estimating photometric redshifts (z-phot).
        
        Expertise Note: 
            The x-axis covers the 'Optical to Mid-Infrared' bridge. The jump from 
            z-band (9134Å) to W1 (33680Å) is critical for seeing the 'Old Stellar 
            Population' bump, which helps distinguish distant galaxies from 
            local red stars.
        
        Args:
            example (Dict): A dictionary containing 'FLUX_X' keys for G, R, I, Z 
                and WISE (W1-W4) bands, plus an 'object_id'.
        """
        # Central wavelengths in Angstroms: g, r, i, z, W1, W2, W3, W4
        wavelengths = [4770, 6231, 7625, 9134, 33680, 46180, 120820, 221940]
        flux_keys = ['FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']
        
        try:
            fluxes = [example[k] for k in flux_keys]
        except KeyError as e:
            print(f"Error: Missing flux key {e} in provided example.")
            return

        plt.figure(figsize=(10, 5))
        plt.scatter(wavelengths, fluxes, color='crimson', zorder=5, s=60)
        plt.plot(wavelengths, fluxes, linestyle='--', alpha=0.4, color='black')

        plt.xscale('log') 
        plt.xlabel('Observed Wavelength (Å) [Log Scale]')
        plt.ylabel('Flux (nanomaggies)')
        plt.title(f"Photometric SED: {example.get('object_id', 'N/A')}")
        plt.grid(True, which="both", ls="-", alpha=0.1)
        plt.show()

    def plot_spatial_grasp(self, example: Dict[str, Any]):
        """
        Compares the RGB source image against its semantic segmentation mask.
        
        Use Case: 
            Debugging source extractor performance and checking for 'deblending' 
            issues where two close stars/galaxies might be treated as one.
        
        Expertise Note: 
            The 'Object Mask' defines the pixels used to calculate the fluxes 
            shown in the SED. If the mask is too small (under-segmentation), 
            your SED fluxes will be underestimated.
            
        Args:
            example (Dict): Dictionary containing 'rgb' and 'object_mask' keys.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # RGB Image
        img = example['rgb']
        if img.shape[0] < img.shape[2]: img = np.transpose(img, (1, 2, 0))
        ax[0].imshow(img)
        ax[0].set_title("Postage Stamp (RGB)")

        # Object Mask
        mask = example['object_mask']
        ax[1].imshow(mask, cmap='magma') # 'magma' often highlights mask gradients better than gray
        ax[1].set_title("Segmentation Mask")

        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.show()

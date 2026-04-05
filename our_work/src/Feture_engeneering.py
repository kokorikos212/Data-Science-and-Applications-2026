
import numpy as np

def engineer_star_likelihood(example, color_hist, extent_hist):
    """
    Engineers a likelihood score based on 1D probability distributions.
    
    Args:
        example (dict): A single data record.
        color_hist (dict): Pre-calculated PDF for Color.
        extent_hist (dict): Pre-calculated PDF for Extent.
    """
    # 1. Extract raw features
    color = example['FLUX_G'] - example['FLUX_R']
    extent = example['shape_r'] # Half-light radius
    
    # 2. Look up probabilities (simplified binning logic)
    # This represents P(Feature | Class)
    p_color_star = color_hist.get(round(color, 1), 1e-6)
    p_extent_star = extent_hist.get(round(extent, 1), 1e-6)
    
    # 3. Create the engineered feature (Log-Likelihood)
    example['star_probability_score'] = np.log(p_color_star) + np.log(p_extent_star)
    
    return example
#!/usr/bin/env python3
"""
Test script to visualize the structure of interventional training and held-out datasets
"""

import jax.numpy as jnp
from jax import random
import numpy as np
import igraph as ig

from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian

# Set up parameters (smaller for easier visualization)
N_VARS = 5
N_OBSERVATIONS = 20
N_HO_OBSERVATIONS = 10
N_INTERVENTION_SETS = 3
PERC_INTERVENED = 0.4  # 40% of variables (2 out of 5)

print("="*60)
print("INTERVENTIONAL DATA STRUCTURE TEST")
print("="*60)
print(f"Variables: {N_VARS}")
print(f"Training observations per dataset: {N_OBSERVATIONS}")
print(f"Held-out observations per dataset: {N_HO_OBSERVATIONS}")
print(f"Number of intervention sets: {N_INTERVENTION_SETS}")
print(f"Percentage of variables intervened per set: {PERC_INTERVENED*100}%")
print()

# Initialize models
key = random.PRNGKey(42)
key, subk = random.split(key)

graph_model = make_graph_model(n_vars=N_VARS, graph_prior_str='er', edges_per_node=2)
generative_model = DenseNonlinearGaussian(
    n_vars=N_VARS, obs_noise=0.1, hidden_layers=(5,), sig_param=1.0
)

# Generate synthetic data
key, subk = random.split(key)
data = make_synthetic_bayes_net(
    key=subk,
    n_vars=N_VARS,
    graph_model=graph_model,
    generative_model=generative_model,
    n_observations=N_OBSERVATIONS,
    n_ho_observations=N_HO_OBSERVATIONS,
    n_intervention_sets=N_INTERVENTION_SETS,
    perc_intervened=PERC_INTERVENED
)

print("Generated Data Summary:")
print(f"  Ground truth graph shape: {data.g.shape}")
print(f"  Observational training data shape: {data.x.shape}")
print(f"  Observational held-out data shape: {data.x_ho.shape}")
print(f"  Number of intervention sets: {len(data.x_interv)}")
print()

# Show intervention details
print("Intervention Sets:")
for i, (interv_dict, interv_x) in enumerate(data.x_interv):
    print(f"  Set {i+1}: Intervened nodes {list(interv_dict.keys())}, shape {interv_x.shape}")
print()

# Create combined training and held-out datasets (same logic as in ensemble_intrv_experiments.py)

# 1. Start with observational data
all_train_data = [data.x]
all_train_masks = [jnp.zeros_like(data.x, dtype=bool)]

all_ho_data = [data.x_ho]
all_ho_masks = [jnp.zeros_like(data.x_ho, dtype=bool)]

# 2. Process each intervention set
for interv_dict, interv_x in data.x_interv:
    # Split interventional data
    n_train_samples = data.x.shape[0]
    n_ho_samples = data.x_ho.shape[0]
    
    # Regenerate full interventional data before splitting
    key, subk = random.split(key)
    interv_x_full = generative_model.sample_obs(
        key=subk,
        n_samples=n_train_samples + n_ho_samples,
        g=ig.Graph.Adjacency(data.g.tolist()),
        theta=data.theta,
        interv=interv_dict
    )

    interv_x_train = interv_x_full[:n_train_samples]
    interv_x_ho = interv_x_full[n_train_samples:n_train_samples + n_ho_samples]
    
    # Training data and mask
    all_train_data.append(interv_x_train)
    mask_train_interv = jnp.zeros_like(interv_x_train, dtype=bool)
    intervened_nodes = list(interv_dict.keys())
    mask_train_interv = mask_train_interv.at[:, intervened_nodes].set(True)
    all_train_masks.append(mask_train_interv)
    
    # Held-out data and mask
    all_ho_data.append(interv_x_ho)
    mask_ho_interv = jnp.zeros_like(interv_x_ho, dtype=bool)
    mask_ho_interv = mask_ho_interv.at[:, intervened_nodes].set(True)
    all_ho_masks.append(mask_ho_interv)

# 3. Concatenate everything
x_train = jnp.concatenate(all_train_data, axis=0)
mask_train = jnp.concatenate(all_train_masks, axis=0)

x_ho_combined = jnp.concatenate(all_ho_data, axis=0)
mask_ho_combined = jnp.concatenate(all_ho_masks, axis=0)

print("="*60)
print("FINAL COMBINED DATASETS")
print("="*60)

print(f"Training data shape: {x_train.shape}")
print(f"Training mask shape: {mask_train.shape}")
print(f"Held-out data shape: {x_ho_combined.shape}")
print(f"Held-out mask shape: {mask_ho_combined.shape}")
print()

print("Training Data (x_train):")
print("First 5 rows of each section:")

# Show observational training section
obs_end = N_OBSERVATIONS
print(f"\n  Observational section (rows 0-{obs_end-1}):")
print(f"    Shape: {x_train[:obs_end].shape}")
print(f"    Sample values:\n{np.array(x_train[:5])}")
print(f"    Corresponding mask (should be all False):\n{np.array(mask_train[:5])}")

# Show each interventional training section
start_idx = obs_end
for i, (interv_dict, _) in enumerate(data.x_interv):
    end_idx = start_idx + N_OBSERVATIONS
    print(f"\n  Intervention set {i+1} section (rows {start_idx}-{end_idx-1}):")
    print(f"    Intervened nodes: {list(interv_dict.keys())}")
    print(f"    Shape: {x_train[start_idx:end_idx].shape}")
    print(f"    Sample values:\n{np.array(x_train[start_idx:start_idx+5])}")
    print(f"    Corresponding mask:\n{np.array(mask_train[start_idx:start_idx+5])}")
    
    # Verify that intervened columns are indeed clamped to 0.0
    intervened_cols = list(interv_dict.keys())
    intervened_values = x_train[start_idx:end_idx, intervened_cols]
    print(f"    Intervened columns {intervened_cols} values (should be all 0.0):")
    print(f"    Min: {jnp.min(intervened_values)}, Max: {jnp.max(intervened_values)}")
    
    start_idx = end_idx

print("\n" + "="*60)
print("HELD-OUT DATA")
print("="*60)

print("Held-out Data (x_ho_combined):")
print("First 5 rows of each section:")

# Show observational held-out section
obs_ho_end = N_HO_OBSERVATIONS
print(f"\n  Observational section (rows 0-{obs_ho_end-1}):")
print(f"    Shape: {x_ho_combined[:obs_ho_end].shape}")
print(f"    Sample values:\n{np.array(x_ho_combined[:5])}")
print(f"    Corresponding mask (should be all False):\n{np.array(mask_ho_combined[:5])}")

# Show each interventional held-out section
start_idx = obs_ho_end
for i, (interv_dict, _) in enumerate(data.x_interv):
    end_idx = start_idx + N_HO_OBSERVATIONS
    print(f"\n  Intervention set {i+1} section (rows {start_idx}-{end_idx-1}):")
    print(f"    Intervened nodes: {list(interv_dict.keys())}")
    print(f"    Shape: {x_ho_combined[start_idx:end_idx].shape}")
    print(f"    Sample values:\n{np.array(x_ho_combined[start_idx:start_idx+5])}")
    print(f"    Corresponding mask:\n{np.array(mask_ho_combined[start_idx:start_idx+5])}")
    
    # Verify that intervened columns are indeed clamped to 0.0
    intervened_cols = list(interv_dict.keys())
    intervened_values = x_ho_combined[start_idx:end_idx, intervened_cols]
    print(f"    Intervened columns {intervened_cols} values (should be all 0.0):")
    if intervened_values.size > 0:
        print(f"    Min: {jnp.min(intervened_values)}, Max: {jnp.max(intervened_values)}")
    else:
        print("    (No held-out samples for this intervention set)")
    
    start_idx = end_idx

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"Total training samples: {x_train.shape[0]}")
print(f"  - Observational: {N_OBSERVATIONS}")
print(f"  - Interventional: {N_INTERVENTION_SETS * N_OBSERVATIONS}")

print(f"\nTotal held-out samples: {x_ho_combined.shape[0]}")
print(f"  - Observational: {N_HO_OBSERVATIONS}")
print(f"  - Interventional: {N_INTERVENTION_SETS * N_HO_OBSERVATIONS}")

print(f"\nPercentage of training samples that are interventional: {100 * (1 - N_OBSERVATIONS / x_train.shape[0]):.1f}%")
print(f"Percentage of held-out samples that are interventional: {100 * (1 - N_HO_OBSERVATIONS / x_ho_combined.shape[0]):.1f}%")

print(f"\nIntervention mask statistics:")
print(f"  Training mask - True entries: {jnp.sum(mask_train)} / {mask_train.size}")
print(f"  Held-out mask - True entries: {jnp.sum(mask_ho_combined)} / {mask_ho_combined.size}")

print("\nDone!") 
"""
Ensemble Experiments for Causal Discovery with Interventional Data (Principled Approach)

This script compares SVGD and Deep Ensembles on their ability to recover
a causal graph when trained and evaluated on a mix of observational and
single-target interventional data.
"""

import jax
import jax.random as random
import jax.tree_util
import numpy as np
import time
import functools
import jax.numpy as jnp
import igraph as ig

from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs.utils import visualize_ground_truth

def create_interventional_data(key, n_vars, n_observations, n_ho_observations, n_intervention_sets, perc_intervened):
    """
    Generates and processes synthetic data with interventions.
    """
    print("\n" + "="*70)
    print("1. GENERATING GROUND TRUTH DATA (WITH SINGLE-TARGET INTERVENTIONS)")
    print("="*70)

    key, subk = random.split(key)

    # Define graph model, and separate generative/likelihood models
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str="sf")

    # This model creates the ground truth data
    generative_model = DenseNonlinearGaussian(
        n_vars=n_vars, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)

    # This model is used by the inference algorithm
    likelihood_model = DenseNonlinearGaussian(
        n_vars=n_vars, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)

    data = make_synthetic_bayes_net(
        key=subk,
        n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        n_intervention_sets=n_intervention_sets,
        perc_intervened=perc_intervened
    )

    # Create the training dataset by combining observational and interventional training data

    # 1. Start with observational training data and a corresponding all-False mask
    all_train_data = [data.x]
    all_train_masks = [jnp.zeros_like(data.x, dtype=bool)]

    # 2. Start with observational held-out data and a corresponding all-False mask
    all_ho_data = [data.x_ho]
    all_ho_masks = [jnp.zeros_like(data.x_ho, dtype=bool)]

    # 3. Process each intervention set: split into training and held-out portions
    for interv_dict, interv_x in data.x_interv:
        # Split the interventional data into training and held-out (same ratio as observational)
        n_train_samples = data.x.shape[0]  # Same as N_OBSERVATIONS
        n_ho_samples = data.x_ho.shape[0]  # Same as N_HO_OBSERVATIONS
        
        key, subk = random.split(key)
        interv_x_full = generative_model.sample_obs(
            key=subk,
            n_samples=n_train_samples + n_ho_samples,
            g=ig.Graph.Adjacency(data.g.tolist()),
            theta=data.theta,
            interv=interv_dict
        )
        
        # Split interventional data
        interv_x_train = interv_x_full[:n_train_samples]
        interv_x_ho = interv_x_full[n_train_samples:n_train_samples + n_ho_samples]
        
        # Add training interventional data
        all_train_data.append(interv_x_train)
        mask_train_interv = jnp.zeros_like(interv_x_train, dtype=bool)
        intervened_nodes = list(interv_dict.keys())
        mask_train_interv = mask_train_interv.at[:, intervened_nodes].set(True)
        all_train_masks.append(mask_train_interv)
        
        # Add held-out interventional data
        all_ho_data.append(interv_x_ho)
        mask_ho_interv = jnp.zeros_like(interv_x_ho, dtype=bool)
        mask_ho_interv = mask_ho_interv.at[:, intervened_nodes].set(True)
        all_ho_masks.append(mask_ho_interv)

    # 4. Concatenate everything into single JAX arrays for JointDiBS
    x_train = jnp.concatenate(all_train_data, axis=0)
    mask_train = jnp.concatenate(all_train_masks, axis=0)

    x_ho_combined = jnp.concatenate(all_ho_data, axis=0)
    mask_ho_combined = jnp.concatenate(all_ho_masks, axis=0)

    print(f"Ground truth graph from a single consistent model: `data.g`")
    print(f"\nTotal training samples: {x_train.shape[0]}")
    print(f"Total held-out samples: {x_ho_combined.shape[0]} (includes interventional)")
    print(f"Percentage of training samples that are interventional: {100 * (1 - data.x.shape[0] / x_train.shape[0]):.2f}%")

    return (x_train, mask_train, x_ho_combined, mask_ho_combined,
            data.g, graph_model, likelihood_model)


# # Setup
key = random.PRNGKey(42)
print(f"JAX backend: {jax.default_backend()}")

# ## 1. Generate Data
(x_train, mask_train, x_ho_combined, mask_ho_combined,
 g_true, graph_model, likelihood_model) = create_interventional_data(
    key=key,
    n_vars=20,
    n_observations=100,
    n_ho_observations=100,
    n_intervention_sets=10,
    perc_intervened=0.1
)


# ## 2. Experiment Parameters
N_PARTICLES = 20
N_ENSEMBLE_RUNS = 20
N_STEPS = 2000


# ## 3. SVGD Baseline (1 run × 20 particles)
print("\n" + "="*70)
print(f"3. SVGD BASELINE (1 run x {N_PARTICLES} particles)")
print("="*70)
key, subk = random.split(key)
dibs_svgd = JointDiBS(x=x_train, interv_mask=mask_train, graph_model=graph_model, likelihood_model=likelihood_model)

start_time = time.time()
gs_svgd, thetas_svgd = dibs_svgd.sample(key=subk, n_particles=N_PARTICLES, steps=N_STEPS)
svgd_time = time.time() - start_time

svgd_empirical = dibs_svgd.get_empirical(gs_svgd, thetas_svgd)
svgd_mixture = dibs_svgd.get_mixture(gs_svgd, thetas_svgd)
print(f"Finished in {svgd_time:.2f}s")


# ## 4. Deep Ensemble (20 runs × 1 particle)
print("\n" + "="*70)
print(f"4. DEEP ENSEMBLE ({N_ENSEMBLE_RUNS} runs x 1 particle)")
print("="*70)
ensemble_gs = []
ensemble_thetas = []

ensemble_start = time.time()
for i in range(N_ENSEMBLE_RUNS):
    print(f"Run {i+1}/{N_ENSEMBLE_RUNS}", end=" ")
    key, subk = random.split(key)
    dibs_single = JointDiBS(x=x_train, interv_mask=mask_train, graph_model=graph_model, likelihood_model=likelihood_model)
    
    gs, thetas = dibs_single.sample(key=subk, n_particles=1, steps=N_STEPS)
    
    ensemble_gs.append(gs)
    ensemble_thetas.append(thetas)
    print("✓")

ensemble_time = time.time() - ensemble_start

# Combine all samples for true ensemble
combined_gs = np.concatenate(ensemble_gs, axis=0)
combined_thetas = jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *ensemble_thetas)

dibs_ensemble = JointDiBS(x=x_train, interv_mask=mask_train, graph_model=graph_model, likelihood_model=likelihood_model)
true_ensemble_empirical = dibs_ensemble.get_empirical(combined_gs, combined_thetas)
true_ensemble_mixture = dibs_ensemble.get_mixture(combined_gs, combined_thetas)
print(f"Finished in {ensemble_time:.2f}s")


# ## 5. Evaluation on Interventional Data

def compute_metrics_interventional(dist, name, dibs_instance, x_ho, mask_ho, g_true):
    """Computes and prints metrics for a given particle distribution."""
    eshd = expected_shd(dist=dist, g=g_true)
    auroc = threshold_metrics(dist=dist, g=g_true)['roc_auc']
    
    # This evaluation function computes the negative log-likelihood on the held-out set.
    def eval_neg_ll(dist, x, mask):
        # vmap over particles
        @functools.partial(vmap, in_axes=(0, 0, None, None))
        def get_ll_samples(g, theta, x, interv_mask):
            return dibs_instance.likelihood_model.log_prob(g=g, theta=theta, x=x, interv_mask=interv_mask)

        ll_samples = get_ll_samples(dist['g'], dist['theta'], x, mask)
        log_probs = dist['log_prob']
        return -jnp.sum(jnp.exp(log_probs) * ll_samples)

    # Evaluate on the held-out data (observational + interventional)
    negll = eval_neg_ll(dist, x_ho, mask_ho)

    print(f'{name:25s} | E-SHD: {eshd:5.2f}  AUROC: {auroc:5.3f}  NegLL: {negll:7.2f}')
    return {'eshd': eshd, 'auroc': auroc, 'negll': negll}

print("\n" + "="*70)
print("5. RESULTS (ON INTERVENTIONAL DATA)")
print("="*70)

# SVGD results
svgd_emp_metrics = compute_metrics_interventional(svgd_empirical, 'SVGD Empirical', dibs_svgd, x_ho_combined, mask_ho_combined, g_true)
svgd_mix_metrics = compute_metrics_interventional(svgd_mixture, 'SVGD Mixture', dibs_svgd, x_ho_combined, mask_ho_combined, g_true)

print("-"*70)

# True ensemble results
true_emp_metrics = compute_metrics_interventional(true_ensemble_empirical, 'Ensemble Empirical', dibs_ensemble, x_ho_combined, mask_ho_combined, g_true)
true_mix_metrics = compute_metrics_interventional(true_ensemble_mixture, 'Ensemble Mixture', dibs_ensemble, x_ho_combined, mask_ho_combined, g_true)
print("="*70)


# ## 6. Summary
print(f"\n" + "="*70)
print("6. SUMMARY")
print("="*70)
print(f"Computation time:")
print(f"  SVGD ({N_PARTICLES} particles):      {svgd_time:6.1f}s")
print(f"  Deep Ensemble ({N_ENSEMBLE_RUNS} × 1):   {ensemble_time:6.1f}s")
print(f"  Time Ratio (Ensemble/SVGD): {ensemble_time/svgd_time:.1f}x")

print(f"\nEmpirical distribution (E-SHD, lower is better):")
print(f"  SVGD:          {svgd_emp_metrics['eshd']:6.2f}")
print(f"  True Ensemble: {true_emp_metrics['eshd']:6.2f}")

print(f"\nEmpirical distribution (AUROC, higher is better):")
print(f"  SVGD:          {svgd_emp_metrics['auroc']:.3f}")
print(f"  True Ensemble: {true_emp_metrics['auroc']:.3f}")

print(f"\nMixture distribution (E-SHD, lower is better):")
print(f"  SVGD:          {svgd_mix_metrics['eshd']:6.2f}")
print(f"  True Ensemble: {true_mix_metrics['eshd']:6.2f}")

print(f"\nMixture distribution (AUROC, higher is better):")
print(f"  SVGD:          {svgd_mix_metrics['auroc']:.3f}")
print(f"  True Ensemble: {true_mix_metrics['auroc']:.3f}")
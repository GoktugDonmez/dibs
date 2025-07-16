
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

from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs.utils import visualize_ground_truth

# # Setup
key = random.PRNGKey(42)
print(f"JAX backend: {jax.default_backend()}")

# ## 1. Generate Ground Truth Data (with Single-Target Interventions)
print("\n" + "="*70)
print("1. GENERATING GROUND TRUTH DATA (WITH SINGLE-TARGET INTERVENTIONS)")
print("="*70)

N_VARS = 20
N_OBSERVATIONS = 100
N_HO_OBSERVATIONS = 100

key, subk = random.split(key)

# Define graph model, and separate generative/likelihood models
graph_model = make_graph_model(n_vars=N_VARS, graph_prior_str="sf")

# This model creates the ground truth data
generative_model = DenseNonlinearGaussian(
    n_vars=N_VARS, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)

# This model is used by the inference algorithm
likelihood_model = DenseNonlinearGaussian(
    n_vars=N_VARS, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)


# Generate observational data first
data_obs = make_synthetic_bayes_net(
    key=subk,
    n_vars=N_VARS,
    graph_model=graph_model,
    generative_model=generative_model, # Use the generative model here
    n_observations=N_OBSERVATIONS,
    n_ho_observations=N_HO_OBSERVATIONS,
    n_intervention_sets=0, # No interventions here
)
print(f"Generated {data_obs.x.shape[0]} observational samples.")

# Now, generate single-target interventional data for one specific target
INTERVENTION_TARGET = 0
key, subk = random.split(key)
# We can reuse the same graph and generative model to ensure the intervention
# happens on the same underlying system.
data_interv = make_synthetic_bayes_net(
    key=subk,
    n_vars=N_VARS,
    graph_model=graph_model,
    generative_model=generative_model, # Use the generative model here
    n_observations=N_OBSERVATIONS,
    n_ho_observations=N_HO_OBSERVATIONS,
    n_intervention_sets=1,
    perc_intervened=0.05 # Corresponds to 1 variable in a 20-variable system
)
interv_dict, x_interv_full = data_interv.x_interv[0]
print(f"Generated {x_interv_full.shape[0]} samples for intervention on target(s): {list(interv_dict.keys())}")

# Create the training and testing datasets with masks
x_train_obs, x_ho_obs = data_obs.x, data_obs.x_ho
x_interv_train, x_interv_ho = np.split(x_interv_full, 2)

mask_interv = np.zeros_like(x_interv_full, dtype=int)
for target_node in interv_dict.keys():
    mask_interv[:, target_node] = 1
mask_interv_train, mask_interv_ho = np.split(mask_interv, 2)

x_train = np.vstack([x_train_obs, x_interv_train])
mask_train = np.vstack([np.zeros_like(x_train_obs, dtype=int), mask_interv_train])
x_ho = np.vstack([x_ho_obs, x_interv_ho])
mask_ho = np.vstack([np.zeros_like(x_ho_obs, dtype=int), mask_interv_ho])

print(f"\nTotal training samples: {x_train.shape[0]}")
print(f"Total held-out samples: {x_ho.shape[0]}")
print(f"Percentage of training samples that are interventional: {100 * (x_interv_train.shape[0] / x_train.shape[0]):.2f}%")


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

def compute_metrics_interventional(dist, name, dibs_instance):
    eshd = expected_shd(dist=dist, g=data_obs.g)
    auroc = threshold_metrics(dist=dist, g=data_obs.g)['roc_auc']
    
    # Create a temporary evaluation function to correctly handle the interventional likelihood.
    def temp_eval_neg_ll(dist, x, mask):
        log_probs = dist.log_prob(g=dist.g, theta=dist.theta)
        ll_fn = functools.partial(dibs_instance.eltwise_log_likelihood_interv, x_ho=x, interv_msk_ho=mask)
        ll_per_particle = jax.vmap(ll_fn)(dist.g, dist.theta)
        ll_samples = jnp.mean(ll_per_particle, axis=1)
        return -jnp.sum(jnp.exp(log_probs) * ll_samples)

    negll = temp_eval_neg_ll(dist, x_ho, mask_ho)

    print(f'{name:25s} | E-SHD: {eshd:5.2f}  AUROC: {auroc:5.3f}  NegLL: {negll:7.2f}')
    return {'eshd': eshd, 'auroc': auroc, 'negll': negll}

print("\n" + "="*70)
print("5. RESULTS (ON INTERVENTIONAL DATA)")
print("="*70)

# SVGD results
svgd_emp_metrics = compute_metrics_interventional(svgd_empirical, 'SVGD Empirical', dibs_svgd)
svgd_mix_metrics = compute_metrics_interventional(svgd_mixture, 'SVGD Mixture', dibs_svgd)

print("-"*70)

# True ensemble results
true_emp_metrics = compute_metrics_interventional(true_ensemble_empirical, 'Ensemble Empirical', dibs_ensemble)
true_mix_metrics = compute_metrics_interventional(true_ensemble_mixture, 'Ensemble Mixture', dibs_ensemble)
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

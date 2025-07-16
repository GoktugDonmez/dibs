
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
from jax import vmap

from dibs.target import make_synthetic_bayes_net, make_graph_model
from dibs.models import DenseNonlinearGaussian
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics

def create_interventional_data(key, n_vars, n_observations, n_ho_observations, n_intervention_sets, perc_intervened):
    """
    Generates synthetic data, combining observational and interventional for training,
    but keeping observational and interventional held-out sets separate.
    """
    print("\n" + "="*70)
    print("1. GENERATING GROUND TRUTH DATA")
    print("="*70)

    key, subk = random.split(key)

    # Define graph model and generative/likelihood models
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str="sf")
    generative_model = DenseNonlinearGaussian(
        n_vars=n_vars, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)
    likelihood_model = DenseNonlinearGaussian(
        n_vars=n_vars, hidden_layers=(5,), obs_noise=0.1, sig_param=1.0)

    # Generate all intervention sets
    data = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars, graph_model=graph_model, generative_model=generative_model,
        n_observations=n_observations, n_ho_observations=n_ho_observations,
        n_intervention_sets=n_intervention_sets, perc_intervened=perc_intervened)

    # --- Create Combined Training Dataset ---
    # Start with observational data
    all_train_data = [data.x]
    all_train_masks = [jnp.zeros_like(data.x, dtype=bool)]

    # Add first (n_intervention_sets-1) interventional datasets for training
    for i in range(n_intervention_sets - 1):
        interv_dict, interv_x_train = data.x_interv[i]
        all_train_data.append(interv_x_train)
        
        # Create intervention mask
        mask_train_interv = jnp.zeros_like(interv_x_train, dtype=bool)
        intervened_nodes = list(interv_dict.keys())
        mask_train_interv = mask_train_interv.at[:, intervened_nodes].set(True)
        all_train_masks.append(mask_train_interv)

    # Finalize training set
    x_train = jnp.concatenate(all_train_data, axis=0)
    mask_train = jnp.concatenate(all_train_masks, axis=0)

    # --- Prepare Held-Out Sets ---
    # Observational held-out (already available)
    x_ho_obs = data.x_ho

    # Interventional held-out (use last intervention set)
    interv_dict_ho, x_ho_intrv = data.x_interv[-1]
    mask_ho_intrv = jnp.zeros_like(x_ho_intrv, dtype=bool)
    intervened_nodes_ho = list(interv_dict_ho.keys())
    mask_ho_intrv = mask_ho_intrv.at[:, intervened_nodes_ho].set(True)

    print(f"Ground truth graph generated.")
    print(f"\nTotal training samples: {x_train.shape[0]} (observational + {n_intervention_sets-1} interventional sets)")
    print(f"Held-out observational samples: {x_ho_obs.shape[0]}")
    print(f"Held-out interventional samples: {x_ho_intrv.shape[0]}")

    return (x_train, mask_train, x_ho_obs, x_ho_intrv, mask_ho_intrv,
            data.g, graph_model, likelihood_model,data_ground_truth)

def compute_metrics(dist, name, dibs_instance, x_ho, mask_ho, g_true):
    """Computes and prints metrics for a given particle distribution."""
    eshd = expected_shd(dist=dist, g=g_true)
    auroc = threshold_metrics(dist=dist, g=g_true)['roc_auc']
    
    # This vmapped function computes the log-likelihood for each particle
    @functools.partial(vmap, in_axes=(0, 0, None, None))
    def get_ll_samples(g, theta, x, interv_mask):
        return dibs_instance.likelihood_model.log_prob(g=g, theta=theta, x=x, interv_mask=interv_mask)

    # Get log-likelihood for each particle in the distribution
    ll_samples = get_ll_samples(dist['g'], dist['theta'], x_ho, mask_ho)
    
    # Compute the posterior predictive log-likelihood, averaged over particles
    log_probs = dist['log_prob']
    negll = -jnp.sum(jnp.exp(log_probs) * ll_samples)

    print(f'{name:25s} | E-SHD: {eshd:5.2f}  AUROC: {auroc:5.3f}  NegLL: {negll:7.2f}')
    return {'eshd': eshd, 'auroc': auroc, 'negll': negll}

# --- SCRIPT START ---
key = random.PRNGKey(42)
print(f"JAX backend: {jax.default_backend()}")

# 1. Generate Data
(x_train, mask_train, x_ho_obs, x_ho_intrv, mask_ho_intrv,
 g_true, graph_model, likelihood_model,data_ground_truth) = create_interventional_data(
    key=key, n_vars=20, n_observations=100, n_ho_observations=100,
    n_intervention_sets=10, perc_intervened=0.1)

# 2. Experiment Parameters
N_PARTICLES = 20
N_ENSEMBLE_RUNS = 20
N_STEPS = 2000

# 3. SVGD Baseline (1 run × 20 particles)
print("\n" + "="*70)
print(f"3. SVGD BASELINE (1 run x {N_PARTICLES} particles)")
print("="*70)
key, subk = random.split(key)
dibs_svgd = JointDiBS(x=x_train, interv_mask=mask_train, graph_model=graph_model, likelihood_model=likelihood_model)

start_time = time.time()
gs_svgd, thetas_svgd = dibs_svgd.sample(key=subk, n_particles=N_PARTICLES, steps=N_STEPS)
svgd_time = time.time() - start_time

svgd_mixture = dibs_svgd.get_mixture(gs_svgd, thetas_svgd)
print(f"Finished in {svgd_time:.2f}s")

# 4. Deep Ensemble (20 runs × 1 particle)
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

combined_gs = np.concatenate(ensemble_gs, axis=0)
combined_thetas = jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *ensemble_thetas)

dibs_ensemble = JointDiBS(x=x_train, interv_mask=mask_train, graph_model=graph_model, likelihood_model=likelihood_model)
true_ensemble_mixture = dibs_ensemble.get_mixture(combined_gs, combined_thetas)
print(f"Finished in {ensemble_time:.2f}s")

# 5. Evaluation
print("\n" + "="*70)
print("5. EVALUATION")
print("="*70)

# --- 5a. On OBSERVATIONAL Held-Out Data ---
print("--- 5a. On OBSERVATIONAL Held-Out Data ---")
mask_ho_obs = jnp.zeros_like(x_ho_obs, dtype=bool) # Mask is all False
svgd_obs_metrics = compute_metrics(svgd_mixture, 'SVGD Mixture (Obs)', dibs_svgd, x_ho_obs, mask_ho_obs, g_true)
ens_obs_metrics = compute_metrics(true_ensemble_mixture, 'Ensemble Mixture (Obs)', dibs_ensemble, x_ho_obs, mask_ho_obs, g_true)

# --- 5b. On INTERVENTIONAL Held-Out Data ---
print("\n--- 5b. On INTERVENTIONAL Held-Out Data ---")
svgd_intrv_metrics = compute_metrics(svgd_mixture, 'SVGD Mixture (Intrv)', dibs_svgd, x_ho_intrv, mask_ho_intrv, g_true)
ens_intrv_metrics = compute_metrics(true_ensemble_mixture, 'Ensemble Mixture (Intrv)', dibs_ensemble, x_ho_intrv, mask_ho_intrv, g_true)

# 6. Summary
print("\n" + "="*70)
print("6. SUMMARY")
print("="*70)
print(f"Computation time:")
print(f"  SVGD ({N_PARTICLES} particles):      {svgd_time:6.1f}s")
print(f"  Deep Ensemble ({N_ENSEMBLE_RUNS} × 1):   {ensemble_time:6.1f}s")

print("\n{:<25} | {:>10} | {:>10}".format("Metric", "SVGD", "Ensemble"))
print("-"*51)
print("{:<25} | {:10.2f} | {:10.2f}".format("E-SHD", svgd_obs_metrics['eshd'], ens_obs_metrics['eshd']))
print("{:<25} | {:10.3f} | {:10.3f}".format("AUROC", svgd_obs_metrics['auroc'], ens_obs_metrics['auroc']))
print("{:<25} | {:10.2f} | {:10.2f}".format("NLL (Observational)", svgd_obs_metrics['negll'], ens_obs_metrics['negll']))
print("{:<25} | {:10.2f} | {:10.2f}".format("NLL (Interventional)", svgd_intrv_metrics['negll'], ens_intrv_metrics['negll']))
print("="*70)

# 7. Save Results
print("\n" + "="*70)
print("7. SAVING RESULTS TO experiment_results.csv")
print("="*70)

results_path = "experiment_results.csv"
with open(results_path, "w") as f:
    f.write("Method,ESHD,AUROC,NLL_Observational,NLL_Interventional,Time_sec\n")
    f.write(f"SVGD,{svgd_obs_metrics['eshd']:.4f},{svgd_obs_metrics['auroc']:.4f},{svgd_obs_metrics['negll']:.2f},{svgd_intrv_metrics['negll']:.2f},{svgd_time:.2f}\n")
    f.write(f"Ensemble,{ens_obs_metrics['eshd']:.4f},{ens_obs_metrics['auroc']:.4f},{ens_obs_metrics['negll']:.2f},{ens_intrv_metrics['negll']:.2f},{ensemble_time:.2f}\n")

print(f"Results successfully saved to {results_path}")

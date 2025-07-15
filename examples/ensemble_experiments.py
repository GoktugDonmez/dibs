# Ensemble Experiments for Causal Discovery

#This script compares two approaches for approximate inference in Bayesian neural networks for causal discovery:
#1. **SVGD (Stein Variational Gradient Descent):** A particle-based variational inference method.
#2. **Deep Ensemble:** A simpler approach that trains multiple models independently with different random initializations.

#The goal is to evaluate their performance on recovering the ground truth causal graph.

import jax
import jax.random as random
import jax.tree_util
import numpy as np
import time

from dibs.target import make_nonlinear_gaussian_model
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs.utils import visualize_ground_truth

# # Setup
key = random.PRNGKey(42)
print(f"JAX backend: {jax.default_backend()}")


# ## 1. Generate Ground Truth Data
print("\n" + "="*70)
print("1. GENERATING GROUND TRUTH DATA")
print("="*70)
key, subk = random.split(key)
data, graph_model, likelihood_model = make_nonlinear_gaussian_model(
    key=subk, n_vars=20, graph_prior_str="sf"
)
print(f"Ground truth graph has {np.sum(data.g)} edges")
try:
    visualize_ground_truth(data.g)
except:
    print("Visualization skipped (may not work in all environments)")


# ## 2. Experiment Parameters
N_PARTICLES = 20
N_ENSEMBLE_RUNS = 20
N_STEPS = 2000


# ## 3. SVGD Baseline (1 run × 20 particles)
print("\n" + "="*70)
print(f"3. SVGD BASELINE (1 run x {N_PARTICLES} particles)")
print("="*70)
key, subk = random.split(key)
dibs_svgd = JointDiBS(x=data.x, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)

start_time = time.time()
gs_svgd, thetas_svgd = dibs_svgd.sample(key=subk, n_particles=N_PARTICLES, steps=N_STEPS)
svgd_time = time.time() - start_time

svgd_empirical = dibs_svgd.get_empirical(gs_svgd, thetas_svgd)
svgd_mixture = dibs_svgd.get_mixture(gs_svgd, thetas_svgd)
print(f"Finished in {svgd_time:.2f}s")


# ## 4. Deep Ensemble (20 runs × 1 particle)
#
# Here, we run 20 independent training processes, each with a single particle. 
# This is equivalent to training 20 models with different random seeds. 
# We then combine the samples from all runs to form the "true ensemble" distribution.
print("\n" + "="*70)
print(f"4. DEEP ENSEMBLE ({N_ENSEMBLE_RUNS} runs x 1 particle)")
print("="*70)
ensemble_gs = []
ensemble_thetas = []

ensemble_start = time.time()
for i in range(N_ENSEMBLE_RUNS):
    print(f"Run {i+1}/{N_ENSEMBLE_RUNS}", end=" ")
    key, subk = random.split(key)
    dibs_single = JointDiBS(x=data.x, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
    
    gs, thetas = dibs_single.sample(key=subk, n_particles=1, steps=N_STEPS)
    
    ensemble_gs.append(gs)
    ensemble_thetas.append(thetas)
    print("✓")

ensemble_time = time.time() - ensemble_start

# Combine all samples for true ensemble
combined_gs = np.concatenate(ensemble_gs, axis=0)
combined_thetas = jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *ensemble_thetas)

# Create true ensemble distributions
dibs_ensemble = JointDiBS(x=data.x, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
true_ensemble_empirical = dibs_ensemble.get_empirical(combined_gs, combined_thetas)
true_ensemble_mixture = dibs_ensemble.get_mixture(combined_gs, combined_thetas)
print(f"Finished in {ensemble_time:.2f}s")


# ## 5. Evaluation
#
# We evaluate both methods using the following metrics:
# - **Expected Structural Hamming Distance (E-SHD):** Lower is better.
# - **Area Under ROC Curve (AUROC):** Higher is better.
# - **Negative Log-Likelihood (NegLL):** Lower is better.
#
# We evaluate both the `empirical` and `mixture` distributions produced by DiBS.
def compute_metrics(dist, name, dibs_instance):
    eshd = expected_shd(dist=dist, g=data.g)
    auroc = threshold_metrics(dist=dist, g=data.g)['roc_auc']
    negll = neg_ave_log_likelihood(dist=dist, eltwise_log_likelihood=dibs_instance.eltwise_log_likelihood_observ, x=data.x_ho)
    print(f'{name:25s} | E-SHD: {eshd:5.2f}  AUROC: {auroc:5.3f}  NegLL: {negll:7.2f}')
    return {'eshd': eshd, 'auroc': auroc, 'negll': negll}

print("\n" + "="*70)
print("5. RESULTS")
print("="*70)

# SVGD results
svgd_emp_metrics = compute_metrics(svgd_empirical, 'SVGD Empirical', dibs_svgd)
svgd_mix_metrics = compute_metrics(svgd_mixture, 'SVGD Mixture', dibs_svgd)

print("-"*70)

# True ensemble results
true_emp_metrics = compute_metrics(true_ensemble_empirical, 'Ensemble Empirical', dibs_ensemble)
true_mix_metrics = compute_metrics(true_ensemble_mixture, 'Ensemble Mixture', dibs_ensemble)
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

# Understanding and Using Interventional Data in DiBS

This document explains the concept of interventional data and how to use it within the `dibs` library via the `interv_mask` parameter.

## 1. Observational vs. Interventional Data

In causal discovery, we aim to uncover the causal relationships between variables, represented as a Directed Acyclic Graph (DAG).

-   **Observational Data:** This is data collected by passively observing a system without interfering with it. While useful, it can be difficult to distinguish true causal links from mere correlations (e.g., via confounders). For example, observing that ice cream sales and crime rates are correlated doesn't mean one causes the other; a third factor (hot weather) is a common cause.

-   **Interventional Data:** This is data collected after performing a direct manipulation or "intervention" on one or more variables in the system. This is often called a "do-operation" (e.g., `do(X=c)`), where we force a variable `X` to take on a specific value `c`. By breaking the natural influence of a variable's parents, interventions provide powerful evidence for identifying causal relationships and ruling out spurious correlations.

## 2. The `interv_mask` Parameter

The `dibs` library is designed to handle both observational and interventional data seamlessly. The `interv_mask` parameter is the key to telling the model which data points are interventional.

-   **What it is:** `interv_mask` is a binary matrix (a NumPy or JAX array) with the exact same shape as your data matrix `x` (i.e., `[n_observations, n_vars]`).

-   **How it works:**
    -   A value of `0` at `interv_mask[i, j]` indicates that the `i`-th observation of the `j`-th variable is **observational**.
    -   A value of `1` at `interv_mask[i, j]` indicates that the `i`-th observation of the `j`-th variable is **interventional** (i.e., its value was set manually).

-   **Default Behavior:** If you don't provide an `interv_mask` (or set it to `None`), `dibs` assumes all data is purely observational. This is equivalent to passing an all-zero matrix.

## 3. How to Use `interv_mask` in Your Workflow

Using interventional data involves two steps: generating the data and the mask, and then passing the mask to the `dibs` model.

### Step 1: Generating Interventional Data and the Mask

The `dibs` library itself does not have a dedicated function to *generate* interventional data. You typically need to simulate this process yourself. Here is a conceptual example:

```python
import numpy as np

# Assume a system with 4 variables and 1000 samples
n_obs = 1000
n_vars = 4

# 1. Generate your observational and interventional data
# This is a placeholder for your actual data generation logic
x_observational = np.random.randn(n_obs // 2, n_vars)  # First 500 are observational
x_interventional = np.random.randn(n_obs // 2, n_vars) # Second 500 are interventional

# For the interventional data, we manually set the value of the 2nd variable (index 1)
x_interventional[:, 1] = 5.0  # do(variable_1 = 5.0)

# Combine into a single dataset
x = np.vstack([x_observational, x_interventional])

# 2. Create the corresponding intervention mask
interv_mask = np.zeros_like(x, dtype=int)

# Mark the intervened variable in the second half of the data
interv_mask[500:, 1] = 1

# Now `x` and `interv_mask` are ready to be used.
```

### Step 2: Passing the Mask to DiBS

Once you have your data `x` and your `interv_mask`, you simply pass them to the `JointDiBS` (or `MarginalDiBS`) constructor. The model will automatically use this information to correctly compute the likelihood during inference.

```python
# Continuing the example from above...

# from dibs.inference import JointDiBS
# from dibs.target import make_nonlinear_gaussian_model
# import jax.random as random

# key = random.PRNGKey(0)
# _, graph_model, likelihood_model = make_nonlinear_gaussian_model(key, n_vars=n_vars)

# Initialize DiBS with the data and the intervention mask
dibs = JointDiBS(
    x=x, 
    interv_mask=interv_mask, 
    graph_model=graph_model, 
    likelihood_model=likelihood_model
)

# Proceed with sampling as usual
# gs, thetas = dibs.sample(key, ...)
```

By providing the `interv_mask`, you are giving the DiBS model crucial information that allows it to correctly learn the causal structure, even when some of the system's natural mechanisms have been overridden by interventions.

## 4. Designing Interventional Experiments

The previous sections explained *how* to use interventional data in `dibs`. This section discusses the more strategic question of *what* interventions to perform for maximal benefit.

Since running experiments can be costly, the goal of **optimal experimental design** is to identify the true causal graph with the minimum number of interventions.

### Key Concepts

-   **Markov Equivalence Class (MEC):** From observational data alone, we can only identify a causal graph up to its MEC. An MEC is a set of different DAGs that all imply the same set of conditional independencies. For example, the graphs `A → B → C` and `A ← B ← C` are in the same MEC. Interventions are required to tell them apart.

-   **Information Gain:** The best intervention is one that maximizes our information gain, meaning it resolves the most ambiguity among the possible causal graphs in the current MEC.

### Strategies for Designing Interventions

Several computational strategies exist to choose the most informative interventions:

1.  **Active Learning:** This is an iterative approach:
    a.  Start with the MEC learned from observational data.
    b.  At each step, select the intervention that is expected to prune the largest number of graphs from the MEC.
    c.  Perform the intervention, update the MEC with the new information, and repeat until only one graph remains.

2.  **Bayesian Optimal Experimental Design (BOED):** This method uses a Bayesian framework to maintain a probability distribution over all possible causal graphs. It then selects the intervention that maximizes the *expected information gain* about this distribution. BOED is powerful but can be computationally expensive.

3.  **Randomized Experiments:** In some scenarios, simply choosing intervention targets at random can be surprisingly effective and has theoretical guarantees for certain graph types.

### How to Choose the Intervention Value (`do(X=c)`)

Once you have chosen *which* variable to intervene on (the target), you must decide *what* value to set it to.

-   **Hard vs. Soft Interventions:**
    -   **Hard Intervention:** This forces a variable to a specific value (e.g., setting a gene's expression to 0). This is the standard `do(X=c)` operation.
    -   **Soft Intervention:** This merely influences the variable, for example, by shifting its mean. This is sometimes more realistic in biological or social systems.

-   **Choosing the Value `c`:**
    -   **Out-of-distribution values:** Setting a variable to a value far from its typical observational range can be very informative, as it can reveal non-linear relationships that might not be apparent otherwise.
    -   **Domain Knowledge:** The choice of `c` is often best guided by domain expertise. For example, in biology, you might set a gene's expression to a value that is known to be biologically significant (e.g., completely knocking it out).
    -   **Random Probing:** If no domain knowledge is available, you could try setting the value to different quantiles of the variable's observed distribution (e.g., the 10th and 90th percentiles) to probe its effect at different levels.


## 5. The Complete Interventional Workflow: Training and Evaluation

The `dibs` library is designed to leverage interventional data throughout the entire machine learning lifecycle: from training to evaluation. The key is to always align the type of data used for training with the type of data used for evaluation.

### The Guiding Principle
> If you train with interventional data, you should evaluate with interventional data. If you train with purely observational data, you should evaluate with purely observational data.

### How the `interv_mask` is Used in Training

The intervention mask is **not** just for evaluation on held-out data. It is a critical component of the training process itself.

When you call `dibs.sample()`, the model begins the SVGD optimization. In every single step, it computes the gradient of the log-probability of the training data. The `interv_mask` is used in this calculation to tell the model which variables had their values forced. For an intervened variable, the model knows not to use that data point to learn the influence of its causal parents, leading to a more accurate gradient and, ultimately, a better model.

### The Two Modes of Evaluation

The `JointDiBS` class provides two distinct methods for calculating likelihood, which you can use for different evaluation goals.

1.  `eltwise_log_likelihood_observ`:
    -   **What it does:** Calculates the likelihood of data assuming it is **all observational**. It does this by internally using an all-zero intervention mask, regardless of the true nature of the data.
    -   **When to use it:** Use this to assess how well your model generalizes to new, unseen **observational** data. This is the method used in all the current library examples.

2.  `eltwise_log_likelihood_interv`:
    -   **What it does:** Calculates the likelihood of data using a specific intervention mask that you provide. It correctly distinguishes between observational and interventional samples.
    -   **When to use it:** Use this to assess how well your model has learned the **true causal mechanisms** of the system, tested with the powerful evidence of interventions.

### Step-by-Step Interventional Experiment Blueprint

Here is a complete blueprint for running a robust interventional experiment with `dibs`.

#### Step 1: Generate and Split the Data

First, create a dataset that contains a mix of observational and interventional samples. Then, split it into training and held-out (testing) sets.

```python
# 1. Generate your full dataset and the corresponding mask
x_full, mask_full = generate_my_interventional_dataset(...)

# 2. Split into training and held-out set   s
# Ensure both sets contain a representative mix of data
x_train, x_ho = x_full[:800], x_full[800:]
mask_train, mask_ho = mask_full[:800], mask_full[800:]
```

#### Step 2: Train the Model on Interventional Data

Instantiate `JointDiBS` with the **training data and training mask**. This ensures the model learns from the interventions.

```python
# 3. Initialize DiBS with the TRAINING data and TRAINING mask
dibs = JointDiBS(
    x=x_train,
    interv_mask=mask_train,
    graph_model=graph_model,
    likelihood_model=likelihood_model
)

# 4. Run the training/sampling process
gs, thetas = dibs.sample(key, ...)
```

#### Step 3: Evaluate the Model on Interventional Data

Use the **held-out data (`x_ho`) and the held-out mask (`mask_ho`)** to evaluate your model's performance. This is where you must use the `eltwise_log_likelihood_interv` method.

```python
# 5. Evaluate the model on the HELD-OUT interventional data
# First, get the posterior distribution of your model's samples
my_distribution = dibs.get_mixture(gs, thetas)

# 6. Call the evaluation function using the `_interv` likelihood method
# Note: The `neg_ave_log_likelihood` function in `dibs.metrics` expects a keyword
# argument `interv_msk_ho` when using an interventional likelihood function.
# This is a conceptual example, and you may need to adapt the call.
# A wrapper function might be needed if the metric function doesn't directly accept the mask.

# Conceptual evaluation call
negll = neg_ave_log_likelihood(
    dist=my_distribution,
    eltwise_log_likelihood=dibs.eltwise_log_likelihood_interv,
    x=x_ho,
    interv_msk_ho=mask_ho  # You would need to pass the mask to the likelihood function
)

print(f"Negative Log-Likelihood on Held-Out Interventional Data: {negll}")
```
**Note:** The `neg_ave_log_likelihood` function in `dibs.metrics` might need a small wrapper to accept and pass the `interv_msk_ho` to the `eltwise_log_likelihood_interv` function, as the base metric function signature is generic.

By following this workflow, you can fully leverage the capabilities of the `dibs` library to both train on and evaluate with interventional data, leading to a much more rigorous and insightful causal analysis.

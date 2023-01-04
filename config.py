env_name = "HalfCheetahVel-v2"
# env_name = "HalfCheetahDir-v2"

env_kwargs = {
    "low": 0.0,
    "high": 2.0,
    "normalization_scale": 10.0,
    "max_episode_steps": 100,
}

if env_name == "HalfCheetahDir-v2":
    env_kwargs = {
        "normalization_scale": 10.0,
        "max_episode_steps": 100,
    }

# EPS for avoiding log(0)
epsilon = 1e-15

# Discount factor gamma.
gamma = 0.99

# Discount factor lambda used in "Generalized Advantage Estimation" (GAE).
gae_lambda = 1.0

# If "true", then the first order approximation of MAML is applied.
first_order = False

# Policy network
# --------------
# Number of hidden units in each layer.
hidden_sizes = [64, 64]

# Non-linear activation function to apply after each hidden layer.
activation = "tanh"

# Task-specific
# -------------
# Number of trajectories to sample for each task.
fast_batch_size = 20

# Number of gradient steps in the inner loop / fast adaptation.
num_steps = 1

# Step size for each gradient step in the inner loop / fast adaptation.
fast_lr = 0.1


# Optimization
# ------------
# Number of outer-loop updates (ie. number of batches of tasks).
num_batches = 1000

# Number of tasks in each batch of tasks.
meta_batch_size = 40

# TRPO-specific
# -------------
# Size of the trust-region.
max_kl = 1.0e-2

# Number of iterations of Conjugate Gradient.
cg_iters = 10

# Value of the damping in Conjugate Gradient.
cg_damping = 1.0e-5

# Maximum number of steps in the line search.
ls_max_steps = 15

# Ratio to use for backtracking during the line search.
ls_backtrack_ratio = 0.8

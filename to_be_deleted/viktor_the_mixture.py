import pymc3 as pm

# Assume 10 trials and 5 successes out of those trials
# Change these numbers to see how the posterior plot changes
trials = 10
successes = 5

# Set up model context
with pm.Model() as coin_flip_model:
    # Probability p of success we want to estimate
    # and assign Beta prior
    p = pm.Beta("p", alpha=1, beta=1)

    # Define likelihood
    obs = pm.Binomial(
        "obs",
        p=p,
        n=trials,
        observed=successes,
    )

    # Hit Inference Button
    idata = pm.sample()

import arviz as az

az.plot_posterior(idata, show=True)

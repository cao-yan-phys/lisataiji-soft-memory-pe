# Data

`lisataiji_samples_thinned.npz` is a thinned posterior sample for the LISA-Taiji example.

It was derived from the full production chain `emcee_LISA-Taiji_4.npz`, which has shape `(50000, 48, 5)` and is not included here.  The released file keeps the second half of the chain with a step thinning factor of 50, giving 24000 posterior samples.

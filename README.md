# Neural-Generative Models for Heterogeneous Attribute Tabular Data

## About
This repo contains an implementation of a VAE and CSVAE with modified likelihoods to explicitly handle tabular data with heterogenous statistical data types including:

* Binary (bernoulli)
* Categorical (categorical)
* Ordinal (ordinal logit model)
* Bounded Count (binomial)
* Unbounded Count (poisson)
* Positive Real (lognormal)
* Real (normal)

Included also are 5 experiments showcasing how this modified likelihood allows for more realistic modeling of tabular data:

* Simple mixed real-categorical with VAE
* Mixed real-categorical with values dependent on latent binary variable with VAE
* Mixed real-categorical with values dependent on latent binary variable with CSVAE
* Dataset with all 7 attributes with CSVAE
* Real dataset (south german credit) with CSVAE

These models were implemented as a part of a course project for AM226.
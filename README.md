# [Constrained Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763)

This repo requires a modified version of [geomstats](https://github.com/geomstats/geomstats) that adds jax functionality, and a number of other modifications. This can be found [here](https://github.com/oxcsml/geomstats/tree/tmlr).

This repository contains the code for the paper `Diffusion Models for Constrained Domains`. This paper theoretically and practically extends score-based generative modelling (SGM) from Reimannian manifolds to any convex subsets of connected and complete Riemannian manifolds.

SGMs are a powerful class of generative models that exhibit remarkable empirical performance. Score-based generative modelling consists of a “noising” stage, whereby a diffusion is used to gradually add Gaussian noise to data, and a generative model, which entails a “denoising” process defined by approximating the time-reversal of the diffusion. 

## Install

Simple install instructions are:
```
git clone https://github.com/oxcsml/score-sde.git
git clone https://github.com/oxcsml/geomstats.git 
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_exps.txt
GEOMSTATS_BACKEND=jax pip install -e geomstats
pip install -e .
```

- `requirements.txt` contains the core requirements for running the code in the `score_sde` and `riemmanian_score_sde` packages. NOTE: you may need to alter the jax versions here to match your setup.
- `requirements_exps.txt` contains extra dependencies needed for running our experiments, and using the `run.py` file provided for training / testing models. Also contains extra dependencies for using the job scheduling functionality of hydra.
- `requirements_dev.txt` contains some handy development packages.

## Code structure

The bulk of the code for this project can be found in 3 places
- The `score_sde` package contains code to run SDEs on Euclidean space. This code is modified from the code from the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://github.com/yang-song/score_sde).
- The `riemannian_score_sde` package contains code needed to extend the code in `score_sde` to Riemannian manifolds.
- An extended version of [geomstats](https://github.com/oxcsml/geomstats) that adds `jax` support, and a number of other extensions.

### Different classes of models
Most of the models used in this paper can be though of as a pushforward of a simple density under some continuous-time transformation into a more complex density. In code, this is represented by a `score_sde.models.flow.PushForward`, containing a base distribution, and in the simplest case, a time dependent vector field that defines the flow of the density through time.

A `Continuous Normalizing Flow (CNF) [score_sde.models.flow.PushForward]`
samples from the pushforward distribution by evolving samples from the base
measure under the action of the vector field. The log-likelihood is computed by
adding the integral of the divergence of the vector field along the sample path
to the log-likelihood of the point under the base measure. Models are trained by
optimising this log-likelihood of the training data.

`Moser flows [score_sde.models.flow.MoserFlow]` alleviate the expensive
likelihood computation in training using an alternative, cheaper, method of
computing the likelihood. This unfortunately requires a condition on the
pushforward vector field, which is enforced by a regularisation term in the
loss. As a result the cheaper likelihood computation unreliable, and the
sampling must still be done with expensive ODE solutions.

`Score-based Generative Models (SGMs) [score_sde.models.flow.SDEPushForward]`
instead consider a pushforward defined by the time-reversal of a noising
Stochastic Differential Equation (SDE). Instead of relying on likelihood based
training, these models are trained using score matching. The likelihood is
computed by converting the SDE to the corresponding likelihood ODE. While
identical in nature to the likelihood ODE of CNFs/Moser flows, these are
typically easier to solve computationally due the learned vector fields
being less stiff.

Other core pieces of code include:

- `score_sde/models/transform.py` which defines transforms between manifolds and Euclidean space, designed to allow for pushing a Euclidean density onto a manifold.
- `score_sde/models/vector_field.py` which contains various parametrisations of vector fields needed for defining the score functions / vector fields
- `score_sde/sde.py` which defines various SDEs
- `score_sde/losses.py` which contains all the loss functions used
- `score_sde/sampling.py` which provides methods for sampling SDEs
- `score_sde/ode.py` which provides methods for solving ODEs

and their counterparts in `riemannian_score_sde`.

### Model structure
Models are decomposed in three blocks:
- a `base` distribution, with `z ~ base` (a 'prior')
- a learnable diffeomorphic `flow: z -> y` (the flexible component of the model, potentially stochastic as for SGMs)
- a `transform` map `y -> x ∈ M` (if the model is *not* defined on the manifold and needs to be 'projected', else  the model is *Riemannian* and `transform=Id`)
Thus, the generative models are defined as `z -> y -> x`.

## Reproducing experiments
Experiment configuration is handled by [hydra](https://hydra.cc/docs/intro/), a highly flexible `yaml` based configuration package. Base configs can be found in `config`, and parameters are overridden in the command line. Sweeps over parameters can also be managed with a single command.

Jobs scheduled on a cluster using a number of different plugins. We use Slurm, and configs for this can be found in `config/server` (note these are reasonably general but have some setup-specific parts). Other systems can easily be substituted by creating a new server configuration.

The main training and testing script can be found in `run.py`, and is dispatched by running `python main.py [OPTIONs]`.

### Logging
By default we log to CSV files and to [Weights and biases](wandb.ai). To use weights and biases, you will need to have an appropriate `WANDB_API_KEY` set in your environment, and to modify the `entity` and `project` entries in the `config/logger/wandb.yaml` file. The top level local logging directory can be set via the `logs_dir` variable.

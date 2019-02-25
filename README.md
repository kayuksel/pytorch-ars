# Augmented Random Search Experiments in PyTorch

*** Check ars_dataparallel.py for a working unique implementation of Augmented Random Search (ARS), which auto-scales accross several GPU(s), on a toy linear regression problem. The rest of the codes in this repository, incl. some evolutionary attempts over ARS, are currently highly experimental ***

This repository contains a work which is inspired by the paper "Simple random search provides a competitive approach to reinforcement learning" (https://arxiv.org/abs/1803.07055) where they developed a competitive alternative to the gradient-based methods for training deep neural policies in reinforcement-learning (RL).

Such randomized approaches are often able to avoid local minima(s) and better approximate the global minima especially when the resources are not limited (whilst authors have been even able to find optimial policies significantly faster than the state-of-the-art methods in RL using Augmented Random Search).

random_search.py contains a simplified version the algorithm that is presented in the paper. The reason of simplification was to reduce the memory usage that is required by the original one.

ars_multiprocess.py is a version that utilizes multiprocesses to make asynchronous updates on a shared model using the random search technique. This is inspired a bit from A2C and A3C methods.

threaded_ars.py contains a multi-threaded implementation where the models are trained. I have been able to easily train 64 models in-parallel. However, there were some speed issues. The aim of this version was to experiment with evolutionary strategies (described below) on top of augmented random search.

ars_dataparallel.py is the closest implementation that I have done to the original augmented random search method as multiple (eight) directions are sampled for a single model. As opposed to others, my implementation distributes a single model that contains multiple directions in its different branches to perform the direction sampling in parallel rather than sequential. The best branch is cloned to other branches after each update.

My objective is integrating a genetic algorithm on top of the parallelized model training. An early version of this where worse 50% of the population is killed in each epoch is already implemented.

Clones of the so far best model are being trained instead of the members of the population that have been terminated in each epoch. The source of inspirations for the genetic part of this are:

- Tabu-Search (https://en.wikipedia.org/wiki/Tabu_search)

Similar to Tabu-search, many models are trained in parallel by testing random mutations on each of them and then the best performing model is copied over the worse-performing members of the population. Tabu-search also discourages agents from coming back to previously visited solutions. I am rather planning to encourage agents to move away from the current best solution of the population in order to encourage exploration in a resource efficient way.

- Simulated Annealing (https://en.wikipedia.org/wiki/Simulated_annealing)

Similar to Simulated Annealing, solutions improving the loss of a mini-batch are accepted by default in the local search and the worsening solutions are accepted with an adaptive probability based on the difference in the current and worsening losses of the mini-batch (which can also have a difficulty that is increasing over epochs). 

- Particle Swarm Optimization (https://en.wikipedia.org/wiki/Particle_swarm_optimization)


Also, the initial experiments are being done in a supervised-learning setting rather than RL. There is also an importance sampling mechanism available for selecting batches at each epoch similar to:

- Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)

Experimental Ideas: 

- Apply Deep Compression techniques from Han. et. al. to reduce the parameter space for random search during the training. What about using ARS for post-training a model that is pre-trained with gradient descent?

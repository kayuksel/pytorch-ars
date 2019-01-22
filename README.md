# pytorch-ars

This repository contains a work which is inspired by the paper "Simple random search provides a competitive approach to reinforcement learning" (https://arxiv.org/abs/1803.07055) where they developed a competitive alternative to the gradient-based methods for training deep neural policies in reinforcement-learning (RL).

Such randomized approaches are often able to avoid local minima(s) and better approximate the global minima especially when the resources are not limited (whilst authors have been even able to find optimial policies significantly faster than the state-of-the-art methods in RL using Augmented Random Search).

random_search.py contains a simplified version the algorithm that is presented in the paper. The reason of simplification was to reduce the memory usage that is required by the original one (as I have been training a huge network that hardly fits into 8GB memory of RTX 2070 when it is trained with the regular methods.

threaded_ars.py contains a multi-threaded implementation where the models are trained. I have been able to easily train 32 models in-parallel with 4 x RTX 2070. However, there are speed issues.

My objective is integrating a genetic algorithm on top of the parallelized model training. An early version of this where worse 50% of the population is killed in each epoch is already implemented.

Clones of the so far best model are being trained instead of the members of the population that have been terminated in each epoch. The source of inspirations for the genetic part of this are:

- Tabu Search (https://en.wikipedia.org/wiki/Tabu_search)
- Simulated Annealing (https://en.wikipedia.org/wiki/Simulated_annealing)

Currently, only solutions improving the loss of a mini-batch are accepted in the local search. I am planning to accept also worsening solutions with a probability Simulated Annealing. This probability can be decreased over time, or can be adaptive based on the difference in the current and worsening losses of the mini-batch. Tabu search discourages agents from coming back to previously visited solutions. I am rather planning to encourage agents to move away from the current best solution of the population in order to encourage exploration as that would be more resource efficient.

- Particle Swarm Optimization (https://en.wikipedia.org/wiki/Particle_swarm_optimization)


Also, the initial experiments are being done in a supervised-learning setting rather than RL.

Please, check the following repository if you are looking for an implementation of the original ARS method in PyTorch:
https://github.com/alexis-jacq/Pytorch_Policy_Search_Optimizer

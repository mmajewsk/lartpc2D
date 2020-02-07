# Lartpc as Reinforcement Learning, using deep q learning

This repository contains a work towards a solution of lartpc-game (see repo [here](https://github.com/mmajewsk/lartpc_game)) problem with reinforcement learning.
If you don't know what is LARTPC you can find useful information in the resources.
This work is based upon dataset from [deeplearnphysics.org](deeplearnphysics.org).
This might be probably the first problem in High Energy Physics solved with reinforcement learning.

## Motivation

The usual approaches to solving semantic segmentation (Like U-net, graph networkj or GAN) use the entirity picture (2D or 3D).
Thus, on an abstract level is much more closer to simulation of a detector, not the particles themselves.
This approach makes sense as it is neighbouring pixels give additional information about specific pixels.
Also, apart from semantic segmentation, a very desired goal of detector models, are efficient symulations.
Usually performed by monte carlo methods, they can be very time consuming.

This brings us to the Reinforcement Learning, and the idea of modelling a single particle.

## Basic concpet

The basic idea is simple. An agent can move through the pixels, with 8 direction freedom. At start it is spawned in random, non blank pixel.

![](https://imgur.com/0Y3S2nQ.png)

The center pixel sets the visible frame for the agent. Only the 9 pixels within the window are visible to the agent.

![](https://imgur.com/5lGLkco.png)

When the agent moves, it moves its visible frame.
With each step, the agent categorises pixels within the window and writes them to blank canvas of the same size as the source image.
The source image are float values, the result is the categorical assigment of one of 3 classes (including blank pixel).

![](https://imgur.com/3sz2Ilo.png)

## Advatages

The method above is just a basic concept, which could be modified in many different ways.
The main advantage is that this resembles much more the actual process that is happenning in the detector.

Also:
- the process can be iterative (agents can go multiple times through the pixels)
- can be parallelized in some cases (there could be an implementation that uses multiple agents at once)
- can be used for symulation (if we will not only learn the network to decide where to go, but also to forecast what the window will look like in next step)
- can bring back time dimension (if we will add memory to the agent, it could be able to recognise the direction of the traveling particle)
- can be interesting to see what happens with agent on a interaction point (maybe it could spawn more agents - just like in real world physics)
- can be implemented in various ways

## Deep Q learning

![](https://imgur.com/Qm0LoG3.png)

The basic implementation consists of two inputs and two outputs.
The source window, on which the categorisation is based, and also the current state of the canvas.
Current state of the canvas is useful for moving the window in correct direction, and for exploration of uncategorised regions of the image.
The movement decision output decides the direction in which the agent should move, and output is pasted onto the current window on canvas.

## Requirements

```
conda install jupyter conda scipy
pip install sklearn pandas pandas-datareader matplotlib pillow requests h5py
pip install --ignore-installed --upgrade tensorflow-gpu 
```


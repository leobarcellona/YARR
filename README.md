![Logo Missing](logo.png)

**NOTE**: This is a fork of the original repository. This fork implement [PerAct](https://peract.github.io/) and [GNFactor](https://yanjieze.com/GNFactor/) changes to RLBench.
The implemented changes are used to generate the data for the experiments in the paper [DreMa: Dream Manipulation](https://arxiv.org/abs/2412.14957).
I'm not the author of the original repository, I just forked it to implement the changes needed for the experiments in the paper (research purpose only). 

----------------------------

**Note**: Pirate qualification not needed to use this library.

YARR is **Y**et **A**nother **R**obotics and **R**einforcement learning framework for PyTorch.

The framework allows for asynchronous training (i.e. agent and learner running in separate processes), which makes it suitable for robot learning.
For an example of how to use this framework, see my [Attention-driven Robot Manipulation (ARM) repo](https://github.com/stepjam/ARM).

This project is mostly intended for my personal use and facilitate my research.

## Install

Ensure you have [PyTorch installed](https://pytorch.org/get-started/locally/).
Then simply run:
```bash
pip install .
```

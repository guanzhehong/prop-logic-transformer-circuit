# A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning

## Overview

This repository contains a walkthrough of the main experiments in the NeurIPS 2025 paper [A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning](https://arxiv.org/pdf/2411.04105). 

The project is built on the [TransformerLens library](https://github.com/TransformerLensOrg/TransformerLens).

## Getting Started

In this repository, we present a series of Jupyter notebooks that walks through the main experiments in our paper. We hope that the more interactive and "tutorial-like" way of presenting the analysis helps convey deeper insights than only reading the paper. The notebooks are contained in the folder `analysis_walkthrough`.

To get started, you will need to install the required packages for the TransformerLens library. However, in the Jupyter notebooks, we also provide a section for environment setup for your convenience. Moreover, you will need a Huggingface account for downloading the Gemma 2 models which we will conduct analyses on. Additionally, the folder `helpers` contain the patching and attention analysis tools which we use in the notebooks for analyzing the LLMs.


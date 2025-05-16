# Agent-Based Modeling of Cell Behavior (PHYS-230 Project)

This repository contains files for Pablo Curiel and Darwin Martinez's project in PHYS-230 at UC Merced. 

## Biological Background

We had presented on modeling the process of contact inhibition of locomotion (CIL). Mesenchymal-to-Epithelial Transition (MET), another collective cell behavior process, made more sense to me (Pablo), so I focused on this process. A high-level overview of this process is described below. 

#### Mesenchymal-to-Epithelial Transition (MET)

MET is an important biological process where cells transition from migrating semi-freely amongst each other (mesenchymal) to a more collectively aligned migration where cells are affected by the trajectory of neighboring cells (epithelial). After enough collective migration and cell replication, cells can clump up to form "sheets" of cells. This process plays an important role in human development and cancer metastasis. 

## Simulating MET

Our first aim was to develop an agent-based (random walk) model to simulate collective cell behavior during MET. This is completed in the file `rw_model.py`. Initially, the model was the exact same as the random walk model from our Lab 2 notebook. I incrementally updated this file to include MET-like aspects (e.g. random $\to$ directed motion). Currently, the simulation does not form sheets, but I believe this is because I omitted implementing cell replication. 

The goal was to make this file object-oriented and modular to allow for easier customizibility and reproducibility. While I don't think it is terrible, I do acknowledge that some OOP principles went out the window as the due date quickly approached. 

### SimMet Class

In short, this is where the *magic* happens. This class

##

## Running the Code

Implemented and tested on: Windows 11, Python 3.10.16, Conda 23.7.2

## File breakdown



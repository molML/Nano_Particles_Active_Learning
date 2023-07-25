![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)

<h2 id="Title">Machine learning-guided polymeric nanoparticle design</h2>

Ana Ortiz-Perez<sup>1</sup>, Derek van Tilborg<sup>1</sup>, Roy van der Meel, Francesca Grisoni<sup>\*</sup>, Lorenzo Albertazzi<sup>\*</sup>\
<sup>1</sup>These authors contributed equally to this work.\
<sup>\*</sup> Corresponding authors: f.grisoni@tue.nl, l.albertazzi@tue.nl. 

This is the codebase that belongs to the paper: ...doi...

**Abstract**
The enormous design space of multicomponent nanoparticles challenges their efficient development and optimization. High throughput methodologies and data-driven computational approaches are attractive emerging strategies to expedite nanoparticle composition design. Here, we show that microfluidics-based production and high content screening guided by active machine learning can rapidly identify PLGA-PEG nanoparticles with a high degree of uptake in breast cancer cells. To the best of our knowledge, this is the first time that these three technologies have been successfully integrated to optimize a biological response through nanoparticle composition. From a small library of particles, uptake was increased up to 15-fold after two rounds of machine learning guided production and screening. Moreover, the resulting predictive model discerns between low and high nanoparticle uptake in cells and can be used to elucidate composition-function relationships. The proposed platform enables rapid and unbiased nanoparticle design.


![Figure 1](figures/fig1.png?raw=true "Figure1")
*Fig 1. Conceptual overview of the proposed iterative nanoparticle design pipeline*


<!-- Prerequisites-->
<h2 id="Prerequisites">Prerequisites</h2>

The following Python packages are required to run this codebase
- [PyTorch](https://pytorch.org/) (1.12.1)
- [Pyro](http://pyro.ai/) (1.8.4)
- [Pandas](https://pandas.pydata.org/) (1.5.3)
- [Numpy](https://numpy.org/) (1.23.5)
- [XGBoost](https://xgboost.readthedocs.io/) (1.7.3)
- [Scikit-learn](https://scikit-learn.org/) (1.2.1)
- [Scikit-optimize](https://scikit-optimize.github.io/) (0.9.0)


<!-- How to cite-->
<h2 id="How-to-cite">How to cite</h2>

You can currently cite our [pre-print](https://chemrxiv.org/engage/chemrxiv/article-details/...):

Ortiz-Perez *et al.* (2023). Machine learning-guided polymeric nanoparticle design. ChemRxiv.   


<!-- License-->
<h2 id="License">License</h2>

This codebase is under MIT license. For use of specific models, please refer to the model licenses found in the original 
packages.
<h2 id="Title">Machine learning-guided high throughput nanoparticle design</h2>

![repo version](https://img.shields.io/badge/Version-v.%201.0-green)
![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)

**Ana Ortiz-Perez**<sup>1</sup>, **Derek van Tilborg**<sup>1</sup>, **Roy van der Meel**, **Francesca Grisoni**<sup>\*</sup>, **Lorenzo Albertazzi**<sup>\*</sup>\
<sup>1</sup>These authors contributed equally to this work.\
<sup>\*</sup>Corresponding authors: f.grisoni@tue.nl, l.albertazzi@tue.nl. 

This is the codebase that belongs to the paper: ...doi...

**Abstract**\
Designing nanoparticles with desired properties is a challenging endeavor, due to the large combinatorial space and complex structure-function relationships. High throughput methodologies and machine learning approaches are attractive and emergent strategies to accelerate nanoparticle composition design. To date, how to combine nanoparticle formulation, screening, and computational decision-making into a single effective workflow is underexplored. In this study, we showcase the integration of three key technologies, namely microfluidic-based formulation, high content imaging, and active machine learning. As a case study, we apply our approach for designing PLGA-PEG nanoparticles with high uptake in human breast cancer cells. Starting from a small set of nanoparticles for model training, our approach led to an increase in uptake from ~5-fold to ~15-fold in only two machine learning guided iterations, taking one week each. To the best of our knowledge, this is the first time that these three technologies have been successfully integrated to optimize a biological response through nanoparticle composition. Our results underscore the potential of the proposed platform for rapid and unbiased nanoparticle optimization.


![Figure 1](figures/fig_summary.png?raw=true "Figure1")
**Figure 1. Conceptual overview of the proposed iterative nanoparticle design pipeline. (a)** the three key integrated technologies: (1) nanoparticles are formulated by microfluidics-assisted nanoprecipitation by controlling different formulation variables xi, (2) the formulations are screened with high content imaging (HCI) to determine their properties yi (e.g., their uptake in MDA-MB-468 cells, as in this proof of concept), and (3) a machine learning model learns the relationship between nanoparticle formulations (x) and their corresponding property (y), and is used to guide the next cycle. **(b)** Overview of experimental cycle: from microfluidic formulation to formulation selection for the following cycle in five days.



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

Ortiz-Perez *et al.* (2023). Machine learning-guided high throughput nanoparticle design. ChemRxiv.   


<!-- License-->
<h2 id="License">License</h2>

This codebase is under MIT license. For use of specific models, please refer to the model licenses found in the original 
packages.

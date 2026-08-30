title: MAPIE
description: Model Agnostic Prediction Interval Estimator — Conformal Prediction for Python
author: MAPIE Team

![MAPIE Logo](https://mapie.readthedocs.io/en/latest/images/mapie_logo_nobg_cut.png){width=400}

# MAPIE — Model Agnostic Prediction Interval Estimator {#mapie-model-agnostic-prediction-interval-estimator}

**An open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.**

 [![Codecov](https://codecov.io/gh/scikit-learn-contrib/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1)](https://codecov.io/gh/scikit-learn-contrib/MAPIE) [![License](https://img.shields.io/github/license/scikit-learn-contrib/MAPIE)](https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE) [![PyPI](https://img.shields.io/pypi/v/mapie)](https://pypi.org/project/mapie/) [![Python](https://img.shields.io/pypi/pyversions/mapie)](https://pypi.org/project/mapie/) [![Downloads](https://img.shields.io/pypi/dm/mapie)](https://pypistats.org/packages/mapie) [![Conda](https://img.shields.io/conda/vn/conda-forge/mapie)](https://anaconda.org/conda-forge/mapie)

[Get Started ](https://mapie.readthedocs.io/en/latest/content/getting-started/quick-start/) [API Reference ](https://mapie.readthedocs.io/en/latest/api/)

---

🚀 MAPIE in 2026 🚀 New features have been implemented, starting with the application of **risk control** to emerging use cases such as **LLM-as-Judge** and **image segmentation**. In addition, **exchangeability tests** have been introduced to help users verify when MAPIE can be legitimately applied. Also, new **adaptive** conformal prediction methods have been added. Finally, the documentation has been updated with a new design!

🎉 MAPIE in 2025 🎉 MAPIE v1 is live! You're seeing the documentation of this new version, which introduces major changes to the API. Extensive release notes are available in the [documentation](https://mapie.readthedocs.io/en/stable/getting-started/v1-release-notes/). You can switch to the documentation of previous versions using the Read the Docs version menu.

---

![Educational Visual](https://mapie.readthedocs.io/en/latest/images/educational_visual.png){width=500}

Image credits: Cemrecan Yurtman (portrait) and hogrmahmood (zebra-horse hybrid).

## What can MAPIE do? {#what-can-mapie-do}

###  Prediction Intervals & Sets {#prediction-intervals-sets}

Compute **prediction intervals** (regression, time series) or **prediction sets** (classification) using state-of-the-art conformal prediction methods.

MAPIE implements **peer-reviewed algorithms** with **theoretical guarantees** under minimal assumptions, based on Conformal Prediction and Distribution-Free Inference.

[Learn more →](https://mapie.readthedocs.io/en/latest/content/conformal-prediction/) \
 [Browse examples →](https://mapie.readthedocs.io/en/latest/generated/regression/)

###  Risk Control {#risk-control}

**Control prediction errors** for complex tasks: multi-label classification, semantic segmentation, with probabilistic guarantees on precision and recall.

[Learn more →](https://mapie.readthedocs.io/en/latest/content/risk-control/) \
 [Browse examples →](https://mapie.readthedocs.io/en/latest/generated/risk_control/)

###  Model Agnostic {#model-agnostic}

Use **any model** — scikit-learn, TensorFlow, PyTorch — thanks to scikit-learn-compatible wrappers. Part of the **scikit-learn-contrib** ecosystem.

[Get started →](https://mapie.readthedocs.io/en/latest/content/getting-started/quick-start/)

---

##  All Examples {#all-examples}

Explore our gallery of hands-on examples covering all MAPIE use cases:

###  Regression {#regression}

Prediction intervals for regression and time series.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/regression/)

###  Classification {#classification}

Prediction sets for single-label and multi-label classification.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/classification/)

###  Conditional Conformal Prediction {#conditional-conformal-prediction}

Conditional prediction intervals and prediction sets.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/conditional_cp/)

###  Risk Control {#risk-control_1}

Control risks for complex ML tasks with probabilistic guarantees.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/risk_control/)

###  Calibration {#calibration}

Calibrate and evaluate probabilistic predictions.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/calibration/)

###  Exchangeability Testing {#exchangeability-testing}

Test distribution shifts and monitor exchangeability assumptions.

[Browse examples →](https://mapie.readthedocs.io/en/latest/generated/exchangeability_testing/)

###  Educational Notebooks {#educational-notebooks}

Work through guided regression and LLM conformal-prediction exercises, then compare your solutions with the completed versions.

[Open educational notebooks →](https://github.com/scikit-learn-contrib/MAPIE/tree/master/notebooks/educational-content)

---

##  Quick Install {#quick-install}

```
pip install mapie
```

See the [Quick Start](https://mapie.readthedocs.io/en/latest/content/getting-started/quick-start/) for other installation methods, requirements, and a first example.

---

##  Citation {#citation}

If you use MAPIE in your research, please cite the main paper:

> Cordier, Thibault, et al. "Flexible and systematic uncertainty estimation with conformal prediction via the MAPIE library." *Conformal and Probabilistic Prediction with Applications.* PMLR, 2023.

```
@inproceedings{Cordier_Flexible_and_Systematic_2023,
    author = {Cordier, Thibault and Blot, Vincent and Lacombe, Louis and Morzadec, Thomas and Capitaine, Arnaud and Brunel, Nicolas},
    booktitle = {Conformal and Probabilistic Prediction with Applications},
    title = {{Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library}},
    year = {2023}
}
```

You can also cite the ICML workshop manuscript:

> Taquet, Vianney, et al. "MAPIE: an open-source library for distribution-free uncertainty quantification." *arXiv preprint arXiv:2207.12274* (2022).

```
@article{taquet2022mapie,
    title = {MAPIE: an open-source library for distribution-free uncertainty quantification},
    author = {Taquet, Vianney and Blot, Vincent and Morzadec, Thomas and Lacombe, Louis and Brunel, Nicolas},
    journal = {arXiv preprint arXiv:2207.12274},
    year = {2022}
}
```

---

##  Affiliations {#affiliations}

MAPIE has been developed through a collaboration between Capgemini Invent, Quantmetry, Michelin, ENS Paris-Saclay, and with the financial support from Région Île-de-France and Confiance.ai.

---

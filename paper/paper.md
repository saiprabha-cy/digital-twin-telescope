---
title: "Digital Twin Telescope: An End-to-End Simulation Framework for Ground-Based Optical Astronomy"
tags:
  - astronomy
  - simulation
  - digital twin
  - optics
  - atmospheric turbulence
  - machine learning
authors:
  - name: SaiPrabha C Y
    affiliation: 1
affiliations:
  - name: Independent Researcher, India
    index: 1
date: 2025-12-22
bibliography: paper.bib
---

## Summary

Modern astronomical instrumentation increasingly relies on simulation-driven design, calibration, and data analysis. This work presents **Digital Twin Telescope**, a modular Python-based simulation framework that models the complete ground-based astronomical imaging pipeline, integrating optical diffraction, atmospheric turbulence, detector response, synthetic sky generation, and AI-based image reconstruction.

The framework enables end-to-end virtual experimentation, supporting education, early-stage research, and benchmarking of machine-learning-based image reconstruction algorithms using physically realistic synthetic data.

## Statement of Need

Existing astronomical simulation tools often focus on isolated components such as optics, detector response, or sky modeling. Digital Twin Telescope addresses this limitation by unifying multiple physical and computational stages into a single extensible pipeline. This integrated approach enables systematic exploration of how subsystem-level parameters influence final image quality and provides a reproducible environment for algorithm development.

The software is designed for students, researchers, and developers seeking an accessible yet physically grounded simulation platform for optical astronomy and AI-assisted data analysis.

## Software Description

Digital Twin Telescope is implemented in Python and follows a modular architecture. The simulation pipeline includes:
- Optical diffraction and aberration modeling using Fourier optics
- Atmospheric turbulence simulation based on statistical phase screens
- Synthetic sky scene generation from astronomical catalogs
- Detector-level image formation with noise modeling
- AI-based image reconstruction and enhancement
- Interactive visualization dashboard for parameter exploration

Each module can be extended independently, supporting future integration with adaptive optics models and real telescope data.

## Availability and Reproducibility

The source code is openly available on GitHub and archived on Zenodo for long-term accessibility and citation stability. All experiments are reproducible using the provided notebooks and datasets.

## Acknowledgements

The author thanks the open-source scientific Python community for providing the foundational tools used in this work.


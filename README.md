# Teleseis

A Python package for processing teleseismic seismic data, providing tools for coordinate rotations, deconvolution, and visualization of seismic waveforms.

## Description

Teleseis is designed to handle common seismic data processing tasks for teleseismic earthquakes, including:

- Coordinate system rotations (NEZ to RTZ, LQT, and PSVH)
- Receiver function deconvolution with water level stabilization
- Seismic data visualization and comparison
- Depth mapping of receiver functions

The package is built using scientific Python libraries and provides an efficient, vectorized implementation suitable for batch processing of seismic traces.

## Main Modules

### `rotate.py`
Contains functions for rotating three-component seismic data between different coordinate systems:
- `nez_to_rtz()`: Rotates North-South, East-West, Vertical components to Radial, Transverse, Vertical
- `nez_to_lqt()`: Rotates to L (longitudinal), Q (quasi-SV), T (transverse) components
- `nez_to_psvh()`: Rotates to P, SV (vertical polarization), SH (horizontal polarization) components

### `spectral.py`
Provides spectral processing tools:
- `decon()`: Performs frequency-domain deconvolution with water level damping
- `bpfilt()`: Applies zero-phase bandpass Butterworth filtering
- `taper()`: Applies cosine-bell tapering to signal ends

### `plotting.py`
Visualization utilities for seismic data:
- `plot_traces()`: Plots multiple seismic traces with customizable labels
- `compare_traces()`: Side-by-side comparison of seismic traces
- `map_1rf()`: Maps receiver functions to depth
- `powspec()`: Calculates and plots power spectral density

## Installation

### Prerequisites
- Python 3.13.11 or higher
- pip package manager

### Install from source
```bash
git clone https://github.com/jochoacontre01/teleseis.git
cd teleseis
pip install -e .
```

### Dependencies
The package requires the following dependencies (automatically installed):
- matplotlib==3.10.8
- numpy==2.4.4
- obspy==1.5.0
- scipy==1.17.1


## Acknowledgement

This package was developed as part of the EASC 6171 course at Memorial University of Newfoundland, based off existing Matlab code. The implementation is based on established seismic processing techniques and algorithms commonly used in teleseismic receiver function studies.

## Authors

- **Original Matlab code**: Stéphane Rondenay (Proffesor), Department of Earth Science, University of Bergen
- **Python translation**: Jesus Ochoa-Contreras (MSc Geophysics student), Earth Sciences Department, Memorial University of Newfoundland
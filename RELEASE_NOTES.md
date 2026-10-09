# Release Notes

## Version 1.1.0 (April 7, 2026)

### Maintenance and Refactoring Release

Teleseis v1.1.0 focuses on improving package structure, fixing critical import issues, and enhancing project organization for better maintainability and usability.

### Key Improvements

#### Package Structure Reorganization
- **Code Relocation**: Moved source files from `src/` directory to `teleseis/` package directory for standard Python package structure
- **Package Configuration**: Updated `pyproject.toml` to properly discover and include the `teleseis` package
- **Module Initialization**: Added `__init__.py` to establish proper package namespace

#### Bug Fixes
- **Import Typo Fix**: Corrected absolute imports in `plotting.py` from `from spectral import` to `from teleseis.spectral import`
- **Package Finding**: Fixed setuptools package discovery configuration to properly locate the teleseis package

#### Documentation & Branding
- **Project Logo**: Added Teleseis SVG logo for visual branding
- **Updated Acknowledgements**: Enhanced acknowledgements section to properly credit original authors
- **Documentation**: Cleaned up release notes presentation in main documentation

### Dependencies

No changes to dependencies:
- matplotlib==3.10.8
- numpy==2.4.4
- obspy==1.5.0
- scipy==1.17.1

### Migration Guide

If upgrading from v1.0.0:
1. Update import statements if you have absolute imports:
   ```python
   # Old (v1.0.0)
   from spectral import decon
   from rotate import nez_to_rtz
   
   # New (v1.1.0)
   from teleseis.spectral import decon
   from teleseis.rotate import nez_to_rtz
   ```
2. Reinstall the package:
   ```bash
   pip install -e .
   ```

### Compatibility

- Python 3.13.11+
- All previous functionality maintained
- Backward compatible API with updated import paths

---

## Version 1.0.0 (April 6, 2026)

### Initial Stable Release

Teleseis v1.0.0 marks the first stable release of this Python package for teleseismic seismic data processing. This release includes comprehensive tools for seismic waveform analysis, coordinate rotations, and receiver function studies.

### New Features

#### Core Processing Modules
- **Coordinate Rotations**: Complete implementation of seismic coordinate system transformations
  - `nez_to_rtz()`: North-East-Z to Radial-Transverse-Z rotation
  - `nez_to_lqt()`: North-East-Z to Longitudinal-Quasi-SV-Transverse rotation
  - `nez_to_psvh()`: North-East-Z to P-SV-SH wave components rotation

- **Spectral Processing**:
  - `decon()`: Frequency-domain deconvolution with water level stabilization
  - `bpfilt()`: Zero-phase bandpass Butterworth filtering
  - `taper()`: Cosine-bell tapering for signal preprocessing

- **Visualization Tools**:
  - `plot_traces()`: Multi-trace seismic waveform plotting
  - `compare_traces()`: Side-by-side trace comparison
  - `map_1rf()`: Receiver function depth mapping
  - `powspec()`: Power spectral density calculation and plotting

#### Package Infrastructure
- Modern Python packaging with `pyproject.toml`
- Comprehensive type hints throughout the codebase
- Detailed docstrings with parameter descriptions
- Example main script demonstrating package usage

### Technical Improvements

- **Vectorized Operations**: Efficient numpy-based implementations for batch processing
- **Sign Convention Corrections**: Fixed sign conventions in PSVH rotations to match MATLAB/ObsPy standards
- **Plotting Enhancements**: Improved scaling and axis limits for better visualization
- **Code Quality**: Refactored for clarity with proper documentation and type annotations

### Dependencies

- matplotlib==3.10.8
- numpy==2.4.4
- obspy==1.5.0
- scipy==1.17.1

### Bug Fixes

- Corrected sign conventions in PSVH rotation functions
- Fixed scaling issues in plotting functions
- Resolved autoscale problems in trace visualization
- Clarified comments regarding ObsPy-MATLAB sign convention differences

### Documentation

- Comprehensive README with installation and usage instructions
- Detailed function docstrings with parameter and return value descriptions
- Example code demonstrating typical workflow

### Educational Context

This package was developed as part of the EASC 6171 course at Memorial University of Newfoundland, implementing standard teleseismic processing techniques used in receiver function seismology.

### Migration Notes

This is the first stable release. No migration is required for new installations.

---

**Installation:**
```bash
pip install -e .
```

**Quick Start:**
```python
from teleseis.rotate import nez_to_rtz
from teleseis.spectral import decon
from teleseis.plotting import plot_traces

# Your seismic processing code here
```
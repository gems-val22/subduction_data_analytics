<!-- #region -->
# Fingerprinting Subduction Margins Using PCA Profiles: A Data Science Approach to Assessing Earthquake Hazard


This repository contains the code for a project that utilizes Principal Component Analysis (PCA) to evaluate earthquake hazards at subduction margins. The methods and analyses presented here form part of the academic research carried out by Valerie Locher during her MSc and PhD.


## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)
- [References](#references)


## Installation

To set up the project environment using conda, follow these steps:

1. Navigate to the project's root directory:

    ```bash
    cd path/to/subduction_data_analytics
    ```

2. Create a new conda environment from the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

3. Activate the newly created environment:

    ```bash
    conda activate subductionPCA
    ```

4. Install the `subductionPCA` package in editable mode:

    ```bash
    pip install -e .
    ```


## Data

We use margin property data compiled by McLellan and Audet (2020). To assign margin segments a maximum observed magnitude, we use data from the USGS earthquake catalogue and four historical earthquakes as described in Schellart and Rawlinson (2013). 

To run this package, download this data and save it as described here:
- **data/feature_data.csv**: the margin property data (McLellan and Audet, 2020)
- **data/historical_earthquakes.csv**: the historical earthquake data (as described in Schellart and Rawlinson, 2013)
- **data/eq_data/filename.csv**: earthquake catalogue files (USGS, 1900-2023). We use a catalogue of wordwide measured earthquakes > M4 as of August 9th 2023. 


## Usage

To use the `subductionPCA` package, start by importing the necessary modules:
<!-- #endregion -->

```python
from subductionPCA import preprocessing, Projector
```
<!-- #region -->
Please ensure the relevant data are downloaded and saved in the correct location, as described above. For detailed examples on how to utilize the package, please refer to the demo.ipynb Jupyter notebook included in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Credits

This project is part of the academic work of Valerie Locher, conducted under the supervision of Rebecca Bell, Cédric John, and Parastoo Salah during her MSc and PhD studies, which are funded by the Department of Earth Science and Engineering’s scholarship in computational science and engineering for women. The code presented here was written using the python library scikit-learn. The empirical relationships relating earthquake magnitude to surface rupture length were derived by Wells and Coppersmith (1994).


## References

McLellan, M., & Audet, P. (2020). Uncovering the physical controls of deep subduction zone slow slip using supervised classification of subducting plate features. Geophysical Journal International, 223(1), 94-110. 

Schellart, W. P., & Rawlinson, N. (2013). Global correlations between maximum magnitudes of subduction zone interface thrust earthquakes and physical parameters of subduction zones. Physics of the Earth and Planetary Interiors, 225, 41-67. 

U.S. Geological Survey. (1900 - 2023). Earthquake Catalog. Retrieved August 09, 2023 from https://earthquake.usgs.gov/earthquakes/search/

Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. Bulletin of the seismological Society of America, 84(4), 974-1002.
<!-- #endregion -->

```python

```

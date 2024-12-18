<!-- #region -->
# Fingerprinting Subduction Margins Using PCA Profiles: A Data Science Approach to Assessing Earthquake Hazard


This repository contains the code for a project that utilizes Principal Component Analysis (PCA) to evaluate earthquake hazards at subduction margins. The methods and analyses presented here form part of the academic research carried out by Valerie Locher during her MSc and PhD.

![alt text](https://github.com/gems-val22/subduction_data_analytics/blob/main/docs/figure_3.png)

PC space relative density maps of segments with assigned maximum magnitudes of 8.5 and above (filled-in contours) and segment distributions for margins with no giant earthquakes in the instrumental record (scatter points) for projections generated using linear PCA, RBF, and cosine Kernel-PCA. 

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)
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

We use margin property data compiled by McLellan and Audet (2020). To assign margin segments a maximum observed magnitude, we use data from the USGS earthquake catalogue and historical earthquakes from the Global Historical Earthquake Archive (Albini et al., 2013). 

To run this package, download this data and save it as described here:
- **data/feature_data.csv**: the margin property data (McLellan and Audet, 2020), access at their [GitHub repository](https://github.com/morgan-mclellan/GJI-S-19-1081) 
- **data/ghea.csv**: Global Historical Earthquake Catalogue - GHEC v1.0, access at https://www.emidius.eu/GEH/ (Albini et al., 2013)
- **data/eq_data/filename.csv**: earthquake catalogue files (USGS, 1900-2023), access at the [USGS earthquake catalogue](https://earthquake.usgs.gov/earthquakes/search/). We use a catalogue of wordwide measured earthquakes > M4 as of August 9th 2023. 


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


## Citation

This work has been submitted for publication in Geology.


## Credits

This work is part of [Valerie Locher](https://github.com/gems-val22)'s MSc and PhD studies, conducted under the supervision of Rebecca Bell, [Cédric John](https://github.com/cedricmjohn), and Parastoo Salah and contributed to by [Robert Platt](https://github.com/rob-platt). Valerie Locher's studies at Imperial College London are funded by the Department of Earth Science and Engineering’s scholarship in computational science and engineering for women. All code from Cédric John's research group can be found in the [John Lab GitHub repository](https://github.com/johnlab-research).

The empirical relationships relating earthquake magnitude to surface rupture length we use here were derived by Wells and Coppersmith (1994). We calculate Wasserstein distances (Wasserstein, 1969) and Adjusted Rand Indices (Hubert and Arabie, 1985).

<br>
<br>

<a href="https://www.john-lab.org">
<img src="https://www.john-lab.org/wp-content/uploads/2023/01/footer_small_logo.png" style="width:220px">
</a>


## References

Albini, P., Musson, R. M. W., Gomez Capera, A. A., Locati, M., Rovida, A., Stucchi, M., & Viganò, D. (2013). Global historical earthquake archive and catalogue (1000-1903). Pavia, Italy.

Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of classification, 2, 193-218. 

McLellan, M., & Audet, P. (2020). Uncovering the physical controls of deep subduction zone slow slip using supervised classification of subducting plate features. Geophysical Journal International, 223(1), 94-110. 

U.S. Geological Survey. (1900 - 2023). Earthquake Catalog. Retrieved August 09, 2023 from https://earthquake.usgs.gov/earthquakes/search/

Wasserstein, L. N. (1969). Markov processes on countable product space describing large systems of automata. Probl. Pered. Inform, 5, 64-73.

Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. Bulletin of the seismological Society of America, 84(4), 974-1002.
<!-- #endregion -->

```python

```

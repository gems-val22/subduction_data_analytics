# Fingerprinting Subduction Margins Using PCA Profiles: A Data Science Approach to Assessing Earthquake Hazard

This repository contains the code for a project that utilizes Principal Component Analysis (PCA) to evaluate earthquake hazards at subduction margins. The methods and analyses presented here form part of the academic research carried out by Valerie Locher during her MSc and PhD.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)

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

## Usage

To use the `subductionPCA` package, start by importing the necessary modules:

```python
from subductionPCA import preprocessing, Projector
```
For detailed examples on how to utilize the package, please refer to the demo.ipynb Jupyter notebook included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits
This project is part of the academic work of Valerie Locher, conducted under the supervision of Professors Rebecca Bell, Cédric John, and Parastoo Salah during her MSc and PhD studies.


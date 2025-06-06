'''

Script to assign maximum magnitudes to subduction margin segments.


Input data needed: 
------------------
- earthquake catalogue (multiple files) in the folder "data/eq_data"
- historical earthquake data (historical_earthquakes.csv) in the folder "data"
- segment location data (feature_data.csv) in the folder "data"

Calls: 
------
- the function eq_binning from binning.py (in the folder "modules") 

'''


import numpy as np
import pandas as pd
from pathlib import Path
import glob
import os

from subductionPCA.binning import eq_binning

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# -------------- EARTHQUAKE DATA: ----------------------------

# importing USGS earthquake catalogue files and combining them:
path = '../data/eq_data'
files = Path(path).glob('*.csv')

dfs = list()
for f in files:
    data = pd.read_csv(f, index_col=None, header=0)
    data['file'] = f.stem 
    dfs.append(data)

eq_data = pd.concat(dfs, ignore_index=True, axis = 0)

# filtering for depth and magnitude measurement and excludeing earthquakes < 4:
eq_data = eq_data[(eq_data['depth'] < 100) | (pd.isnull(eq_data['depth']))] # keeps data points that have no information on depth
eq_data = eq_data[(eq_data.magType == 'mw') | (eq_data.magType == 'mww') | (eq_data.magType == 'mwb') \
                     | (eq_data.magType == 'mwc') | (eq_data.magType == 'mwr')]
eq_data = eq_data[eq_data['mag'] > 4]

# dropping irrelevant columns (only longitude, latitude, and magnitude are needed)
to_drop = ['time', 'depth', 'nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated', 'place', 'type', 'horizontalError', \
           'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource', 'file']

eq_data = eq_data.drop(columns = to_drop)
eq_data = eq_data.reset_index().drop(columns = 'index') 

# adding historical earthquake data:
historic_eqs = pd.read_csv('../data/ghea.csv')[['M', 'Lon', 'Lat']].rename(columns = {'M':'mag', 'Lon':'longitude', 'Lat':'latitude'})
# historic_eqs = pd.read_csv('../data/historical_earthquakes.csv').drop(columns = ['margin', 'year'])
eq_data = pd.concat([historic_eqs, eq_data])

# discarding earthquakes outside of the Pacific (to decrease the binning algorithm's run time): 
eq_data = eq_data[(eq_data.longitude < -40) | (eq_data.longitude > 60)]


# -------------- SUBDUCTION ZONE PARAMETER DATA ----------------------------

all_data = pd.read_csv('../data/feature_data.csv')

# missing values: change from #NUM! to np.nan
for col in all_data.columns:
    all_data[col] = np.where(all_data[col] == '#NUM!', np.nan, all_data[col])

# correcting a spelling mistake:
all_data.loc[all_data['Sub_Zone'] == 'Hikruangi', 'Sub_Zone'] = 'Hikurangi'

# deleting duplicated locations (latitude / longitude values were read from plots created to show the overlapping margins):
all_data.drop(all_data.loc[all_data.Sub_Zone=='Solomon'][all_data.Longitude > 161].index, inplace=True)
all_data.drop(all_data.loc[all_data.Sub_Zone=='Kuril_Kamchatka']\
              [(all_data.Longitude > 164.1) | (all_data.Longitude < 150)].index, inplace=True)
all_data.drop(all_data.loc[all_data.Sub_Zone=='Izu_Bonin']\
              [(all_data.Latitude > 31) | (all_data.Latitude < 28)].index, inplace=True)



# -------------- ASSIGNING MAXIMUM MAGNITUDES ----------------------------

segment_data = all_data.copy() 

# Assigning maximum magnitudes using the binning module (this may take ca. 1 hr to run):

segment_data = eq_binning(segment_data, eq_data)


# -------------- EXPORTING FINAL DATASET ----------------------------

segment_data.to_csv('../data/preprocessed_data_ghea.csv')


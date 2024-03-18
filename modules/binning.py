'''
Module containing functions to pre-process the earthquake data and bin it to segments.
Imported and called in the preprocessing.py script. 
'''

import numpy as np
import pandas as pd


def calculate_SRL(M):
    '''
    Calculates the surface rupture length (SRL) of an earthquake with a certain magnitude, according to the imperical
    relation found by Wells & Coppersmith (1994) for earthquakes of M > 7.7145; for lower M, returns 100 km. (The
    SRL for M = 7.7145 is 100km.)
        
    Parameters: 
    - M: magnitude of the earthquake
        
    Returns: 
    - SRL: the calculated surface rupture length
    
    Called in: 
    - eq_binning
        '''
    
    a, b = -2.86, 0.63
    
    if M > 7.7145: # 6.28: for this magnitude, SRL is just about 12.5 km 
        SRL = 10**(a + b*M)
    else: 
        SRL = 100 # 12.5 is half of the segment width (along the trench)
        
    return SRL


def degrees_to_radians(angle):
    '''
    Converts an angle in degrees into radians. 
    
    Parameters:
    - angle: The angle in degres
    
    Returns:
    - the angle in radians
    
    Called in: 
    - lon_to_km
    '''
    
    return angle * np.pi / 180


def lat_to_km(delta_lat):
    '''
    Converts a distance degrees latitude into km. 
    
    Parameters:
    - delta_lat: distance in degrees latitude
    
    Returns:
    - the distance in km
    
    Called in: 
    - euclidean_distance
    '''
    
    return delta_lat * 111


def lon_to_km(delta_lon, lat):
    '''
    Converts a distance degrees longitude into km. 
    
    Parameters:
    - delta_lon: distance in degrees longitude
    - lat: the latitude at which the distance is measured
    
    Returns:
    - the distance in km
    
    Calls:
    - degrees_to_radians
    
    Called in: 
    - euclidean_distance
    '''
    
    return delta_lon * 111 * np.cos(degrees_to_radians(lat))


def euclidean_distance(x1, y1, x2, y2):
    '''
    Calculates the euclidean distance between two points (x1,y1) and (x2,y2). 
    
    Inputs: 
    - x1, y1: longitude and latitude of point 1
    - x2, y2: longitude and latitude of point 2
    
    Returns: 
    - distance: the euclidean distance between the two points in km 
    
    Calls: 
    - lon_to_km
    - lat_to_km
    
    Called in: 
    - eq_binning
    '''
    
    avg_lat = np.average([y1, y2])
    
    delta_x = lon_to_km(x2 - x1, avg_lat)
    delta_y = lat_to_km(y2 - y1)
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    return distance


def eq_binning(segment_data, eq_data):
    '''
    Finds the highest-magnitude earthquake for each segment. A magnitude is assigned to a segment if the earthquake occurred
    within a distance of the earthquake's SRL (see calculate_SRL). 
    
    Parameters:
    - segment_data: dataframe containing the coordinates of the segment centres 
    - eq_data: list of earthquakes containing their epicentres' coordinates and their magnitudes
    
    Returns:
    - segment_data: dataframe containing the coordinates of segment centres and their assigned maximum magnitudes
    
    Calls: 
    - calculate_SRL
    - euclidean_distance
    
    Called in: 
    - preprocessing.py
    '''
    
    segment_data['Max_mag'] = np.zeros(segment_data.shape[0])
    
    eq_data = eq_data.sort_values(by = 'mag', ascending = False).reset_index()
    
    eq_data['SRL'] = np.zeros(eq_data.shape[0])
    for i in range(eq_data.shape[0]):
        eq_data['SRL'].iloc[i] = calculate_SRL(eq_data.mag.iloc[i])
    
    for s in range(segment_data.shape[0]): # go through segments
        x1, y1 = segment_data.iloc[s].Longitude, segment_data.iloc[s].Latitude

        for e in range(eq_data.shape[0]): # go through earthquakes until it finds one that matches the segment 
                                          # this will be the maximum magnitude earthquake since we've sorted the eq dataframe

            x2, y2 = eq_data.iloc[e].longitude, eq_data.iloc[e].latitude

            if (x1-10 <= x2) & (x1+10 >= x2) & (y1-10 <= y2) & (y1+10 >= y2): # this is so that the algorithm have to do the 
                                                                              # euclidean distance calculation for all data 
                distance = euclidean_distance(x1, y1, x2, y2)

                if distance <= eq_data.iloc[e].SRL/2:
                    segment_data['Max_mag'].iloc[s] = eq_data.iloc[e].mag
                    break
                    
    return segment_data

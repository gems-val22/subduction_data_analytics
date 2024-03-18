import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pykrige.ok import OrdinaryKriging
from scipy.stats import gaussian_kde


def heatmap_diff(fig, ax, heatmap_data, threshold = 8.5, legend_loc = 'right', fs = 11, excl_age = False, cbar = True):
    
    xmin, xmax = heatmap_data.PC1.min()-0.1, heatmap_data.PC1.max()+0.1
    ymin, ymax = heatmap_data.PC2.min()-0.1, heatmap_data.PC2.max()+0.1
        
    # calculate overall density in the PC space
    x1, y1 = np.array(heatmap_data.PC1, dtype = float), np.array(heatmap_data.PC2, dtype = float)
    xi, yi = np.mgrid[xmin:xmax:x1.size**0.5*1j, ymin:ymax:y1.size**0.5*1j]
    k1 = gaussian_kde(np.vstack([x1, y1]))
    zi1 = k1(np.vstack([xi.flatten(), yi.flatten()]))
    
    # calculate density of M8.5+ earthquakes in the PC space
    heatmap_data = heatmap_data[heatmap_data.Max_mag >= threshold]
    x2, y2 = np.array(heatmap_data.PC1, dtype = float), np.array(heatmap_data.PC2, dtype = float)
    k2 = gaussian_kde(np.vstack([x2, y2]))
    zi2 = k2(np.vstack([xi.flatten(), yi.flatten()]))
    
    zi = zi2-zi1

    heatmap = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap = 'coolwarm', alpha = 0.7)

    pc_axis_labels(ax, xmin, xmax, ymin, ymax, fs = fs)
    
    if cbar == True:
        cbar = fig.colorbar(heatmap, ax = ax, location = legend_loc)
        cbar.set_label(label=f'Magnitude $\geq$ {threshold} density difference', fontsize = fs)
        cbar.ax.tick_params(labelsize=fs)
    
    return heatmap


def pc_space_distribution(ax, data, legend = True, fs = 11, axes_adjust = True):
    
    zones, zone_color_dict, zone_label_dict = get_zone_dicts()

    for zone in zones:
        if zone in data.Sub_Zone.unique():
            zonedata = data[data.Sub_Zone == zone]
            ax.scatter(zonedata.PC1, zonedata.PC2, c = zone_color_dict[zone], s = 10, label = zone_label_dict[zone], alpha = .5)
            
        else: 
            pass
        
    if legend == True:
        lgnd = ax.legend(loc = (0, -.33), ncol = 5, fontsize = fs)
        for i in range(len(data.Sub_Zone.unique())):
            lgnd.legendHandles[i]._sizes = [40]
    
    if axes_adjust == True:
        xmin, xmax = data.PC1.min()-0.1, data.PC1.max()+0.1
        ymin, ymax = data.PC2.min()-0.1, data.PC2.max()+0.1
        pc_axis_labels(ax, xmin, xmax, ymin, ymax, fs = fs)
    

def get_low_margin_data(data):
    below = []

    for zone in data.Sub_Zone.unique():
        zonedata = data[data.Sub_Zone == zone]
        if max(zonedata.Max_mag) < 8.5:
            below.append(zone)

    return data[data.Sub_Zone.isin(below)]


def mag_density(data, ax, threshold = 8.5, legend = False):
    
    xmin, xmax = 1.25*data.PC1.min(), 1.25*data.PC1.max()
    ymin, ymax = 1.25*data.PC2.min(), 1.25*data.PC2.max()
    
    high_mag_data = data[data['Max_mag'] >= threshold]
    
    x = high_mag_data.PC1
    y = high_mag_data.PC2

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(np.array(values, dtype = 'float64'))
    f = np.reshape(kernel(positions).T, xx.shape)
    
    cset = ax.contour(xx, yy, f, colors='k', levels = [.3, .45, .6])

    ax.clabel(cset, inline=1, fontsize=10)
    txt = '$M_{max}$'
    
    if legend == True:
        handles = [mpatches.Patch(edgecolor='k', facecolor = 'none', label= f'Density of segments \n with {txt} $\geq$ {threshold}')]
        ax.legend(handles=handles) 
        pc_axis_labels(ax, xmin, xmax, ymin, ymax)
        
    else: 
        return txt, threshold

    
def mag_range_density(data, ax, mag_range = [7,8.5], threshold = 8.5, legend = False):
    
    xmin, xmax = 1.25*data.PC1.min(), 1.25*data.PC1.max()
    ymin, ymax = 1.25*data.PC2.min(), 1.25*data.PC2.max()
    
    mag_range_data = data[(data['Max_mag'] >= mag_range[0]) & (data['Max_mag'] < mag_range[1])]
    
    x = mag_range_data.PC1
    y = mag_range_data.PC2

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(np.array(values, dtype = 'float64'))
    f = np.reshape(kernel(positions).T, xx.shape)
    
    cset = ax.contour(xx, yy, f, linestyles = 'dashed', colors='k', levels = [.3,.45,.6])

    ax.clabel(cset, inline=1, fontsize=10)
    txt = '$M_{max}$'
    
    if legend == True:
        handles = [mpatches.Patch(edgecolor='k', facecolor='none', linestyle='--', \
                         label = f'Density of segments with {mag_range[0]} $\leq$ {txt} < {mag_range[1]}')]
        ax.legend(handles=handles) 
        pc_axis_labels(ax, xmin, xmax, ymin, ymax)
        
    else: 
        return txt, mag_range
    
    

def feature_kriging(data, feature):
    
    OK = OrdinaryKriging(
            data['PC1'], 
            data['PC2'], 
            data[feature], 
            variogram_model='power',
            verbose=False,
            enable_plotting=False,
            nlags=30,
        )
    
    x = np.linspace(data.PC1.min()*1.25, data.PC1.max()*1.25, 50)   
    y = np.linspace(data.PC2.min()*1.25, data.PC2.max()*1.25, 50)

    feature_kriged, var = OK.execute("grid", x, y)
    
    return feature_kriged, var, x, y


def plotting_kriging_map(data, feature, ax):
    
    feature_kriged, _, x, y = feature_kriging(data, feature)
    
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    vmin, vmax = data[feature].min(), np.percentile(data[feature], 90)
    
    cax = ax.imshow(feature_kriged, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap = 'coolwarm', vmin=vmin, vmax=vmax)
    
    ax.scatter(data['PC1'], data['PC2'], s = 60, c = 'white')
    ax.scatter(data['PC1'], data['PC2'], s = 40, c = data[feature], cmap = 'coolwarm', alpha = .5,  vmin=vmin, vmax=vmax)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    feature_dict, unit_dict = get_feature_dicts()
    ax.set_title(feature_dict[feature], size = 14)
    
    return cax 



def pc_axis_labels(ax, xmin, xmax, ymin, ymax, fs = 11):
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('PC1', fontsize = fs)
    ax.set_ylabel('PC2', fontsize = fs)
    
    
def get_feature_dicts():
    
    feature_dict = {'Sed_Thick': 'Sediment thickness', 'Age': 'Plate Age', 'Dip': 'Dip angle',\
                'Vel': 'Relative plate velocity', 'Rough': 'Roughness'}
    
    unit_dict = {'Sed_Thick': 'm', 'Age': 'Ma', 'Dip': '°', 'Vel': 'mm/yr', 'Rough': 'mGal'}
    return feature_dict, unit_dict


def get_zone_dicts():
    
    zones = ['Sumatra', 'Solomon', 'Vanuatu', 'Tonga_Kermadec', 'Hikurangi', 'Kuril_Kamchatka', 'Japan', 'Izu_Bonin', \
         'Mariana', 'Nankai_Ryuku',  'Alaska_Aleutian', 'Cascadia', 'Middle_America', 'South_America']
    
    zone_color_dict = {'South_America': 'midnightblue', 'Sumatra': 'olive', 'Alaska_Aleutian': 'darkgreen', \
                       'Middle_America': 'r', 'Tonga_Kermadec': 'tab:brown', 'Mariana':'lawngreen', \
                       'Nankai_Ryuku': 'tab:cyan', 'Vanuatu': 'tab:purple', 'Solomon': 'deeppink', 'Japan': 'b', \
                       'Cascadia': 'gold', 'Kuril_Kamchatka': 'tab:gray', 'Hikurangi': 'tab:orange', 'Izu_Bonin': 'k'}

    zone_label_dict = {'South_America': 'South America', 'Sumatra': 'Sumatra', 'Alaska_Aleutian': 'Alaska-Aleutian', \
                       'Middle_America': 'Middle America', 'Tonga_Kermadec': 'Tonga-Kermadec', 'Mariana':'Mariana', \
                       'Nankai_Ryuku': 'Nankai-Ryukyu', 'Vanuatu': 'Vanuatu', 'Solomon': 'Solomon', 'Japan': 'Japan', \
                       'Cascadia': 'Cascadia', 'Kuril_Kamchatka': 'Kuril-Kamchatka', 'Hikurangi': 'Hikurangi', \
                       'Izu_Bonin': 'Izu-Bonin'}
    
    return zones, zone_color_dict, zone_label_dict



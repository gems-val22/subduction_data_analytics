'''
Module containing functions for creating figures.
Imported and called by FigureMaker. 
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D



def get_zone_dicts():
    '''
    Defines and returns a list of all considered subduction margins as well as dictionaries containing the margin's full 
    names and their assigned colours for plotting

    Returns: 
    -----------
    - zones: list containing the names of all considered subduction margins
    - zone_color_dict: dictionary containing the margin's names and their assigned colours for plotting
    - zone_label_dict: dictionary containing the margin's names and their better-formatted version 
    '''

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
    
    subtitle_dict = {'Sumatra': 'Quiet and extreme:', 'Solomon': 'Active and moderate:', 'Vanuatu': 'Active and moderate:',
                 'Tonga_Kermadec': 'Active and moderate:','Hikurangi': 'Quiet and extreme:', 
                 'Kuril_Kamchatka': 'Quiet and extreme:','Japan': 'Quiet and extreme:', 
                 'Izu_Bonin': 'Quiet and extreme:', 'Mariana': 'Active and moderate:',
                 'Nankai_Ryuku': 'Quiet and extreme:', 'Alaska_Aleutian': 'Quiet and extreme:',
                 'Cascadia': 'Quiet and extreme:', 'Middle_America': 'Quiet and extreme:',
                 'South_America': 'Quiet and extreme:'}

    return zones, zone_color_dict, zone_label_dict, subtitle_dict


def get_feature_dicts():
    """
    Defines and returns eturns dictionaries mapping the abbrevations of the geological features to their full names and 
    corresponding units.

    Returns:
    -----------
        feature_dict : dict
            Dictionary mapping feature keys to their descriptions.
        unit_dict : dict
            Dictionary mapping feature keys to their units.
    """
    
    feature_dict = {'Sed_Thick': 'Sediment thickness', 'Age': 'Plate Age', 'Dip': 'Dip angle',\
                'Vel': 'Relative plate velocity', 'Rough': 'Roughness'}
    unit_dict = {'Sed_Thick': 'm', 'Age': 'Ma', 'Dip': 'degrees',\
                'Vel': 'mm/yr', 'Rough': 'mGal'}
    return feature_dict, unit_dict


def get_low_margin_data(data):
    """
    Filters and returns data corresponding to margins with no M 8.5+ earthquakes since 1900.

    Parameters:
    -----------
        data (pandas.DataFrame): A DataFrame containing geological data with a 'Sub_Zone' column.

    Returns:
    -----------
        pandas.DataFrame: A filtered DataFrame containing only rows where 'Sub_Zone' matches
                          predefined 'quiet margins'.
    """
    
    quiet_margins = ['Cascadia', 'Hikurangi', 'Izu_Bonin', 'Mariana', 'Middle_America', 'Nankai_Ryuku', \
                      'Solomon', 'Tonga_Kermadec', 'Vanuatu']

    return data[data.Sub_Zone.isin(quiet_margins)]


def get_mag_cmap():
    """
    Creates and returns a colormap and normalization object for maximum magnitude visualization.

    Returns:
    -----------
        cmap : matplotlib.colors.ListedColormap
            A 'viridis' colormap with 11 discrete levels. Values below the range will be colored black ('k').
        norm : matplotlib.colors.Normalize 
            Normalization object for scaling magnitudes between vmin=4 and vmax=9.5. 
    """
    
    norm = plt.Normalize(vmin=4, vmax=9.5)
    cmap = mpl.cm.get_cmap('viridis', 11) 
    cmap.set_under('k')
    return cmap, norm


def create_custom_legend(ax, scatter_handles, fontsize=12, legend_loc = (1.05, 0.1)): 
    """
    Creates a customised legend for scatter plots coloured by subduction margins and groups them into the 
    "active and moderate" and "quiet and extreme" subtitles. 
    
    Parameters:
    -----------
        ax : matplotlib.axes.Axes
            The axes on which the legend will be drawn.
        scatter_handles : dict  
            A dictionary where:
                - Keys (str): Subtitle labels to group scatter plot elements.
                - Values (list): List of scatter plot handles to be included under each subtitle.
        fontsize : int, optional
            Font size for legend text. Default is 12.
        legend_loc : tuple, optional 
            Location of the legend on the plot as (x, y) coordinates. Default is (1.05, 0.1).

    Returns:
    -----------
        lgnd : matplotlib.legend.Legend
            The customized legend object.
    """
    
    handles = []
    labels = []
    
    for subtitle, scatter_list in scatter_handles.items():
        handles.append(Line2D([0], [0], color='none', label=subtitle))
        labels.append(subtitle)
        
        for scatter in scatter_list:
            handles.append(scatter)
            labels.append(scatter.get_label())

    lgnd = ax.legend(handles=handles, labels=labels, loc=legend_loc, ncol=1, fontsize=fontsize, handletextpad=0.5)
    
    for text in lgnd.get_texts():
        if text.get_text() in scatter_handles: # for subtitles
            text.set_fontsize(fontsize)
            text.set_weight('bold')
            text.set_fontstyle('italic')
        else: # for margins
            text.set_fontsize(fontsize)

    lgnd._legend_box.align = "left"
    return lgnd



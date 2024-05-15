import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches

from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde
from pandas.plotting import parallel_coordinates


class FigureMaker():
    
    '''
    Class to create figures to show the PC projections. 
    
    Called in the class Projector (if the input parameter generate_figures is set to True). 
    
    Inputs:
    - data: pandas dataframe containing margin property values and principal components for all segments 
    '''
   

    def __init__(self, data):
        
        self.data = data
        self.xmin, self.xmax, self.ymin, self.ymax = data.PC1.min()*1.25, data.PC1.max()*1.25, data.PC2.min()*1.25, data.PC2.max()*1.25
        
        self.features = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough']
        
        self.pc_space_plots()
        self.pc_magnitude_plots()
        self.feature_distribution_plots()
        self.pc_connections_plots()
        
        
    def pc_space_plots(self):
        ''' 
        Creates plots of the PC projections colour coded by each feature (margin property).
        
        Called when this class is initiated. 
        
        Calls:
        - pc_space_distribution
        - get_low_margin_data
        - heatmap_diff 
        '''
        
        # define figure and subplots
        fig = plt.figure(figsize = (15,5))
        grid = ImageGrid(fig, 111,
                        nrows_ncols = (1,2),
                        axes_pad = .6,
                        label_mode="L",
                        cbar_location = "right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.3)

        # subplot 0: scatter plot the PC space distribution colour coded by margins
        self.pc_space_distribution(grid[0], data = self.data)
        
        # subplot 1: PC space density map for M >= 8.5 segments and PC space distribution of margins with maximum magnitude < 8.5
        heatmap = self.heatmap_diff(grid[1])
        self.pc_space_distribution(grid[1], self.get_low_margin_data())
        
        # plot titles
        grid[0].set_title('PC space by margin', fontsize = 14)
        grid[1].set_title('Hazard assessment', fontsize = 14)
        
        # axes labels and extent
        grid[0].set_ylabel('PC2')
        for i in range(2):
            grid[i].set_xlim([self.xmin, self.xmax])
            grid[i].set_ylim([self.ymin, self.ymax])
            grid[i].set_xlabel('PC1')
        
        # margins legend
        lgnd = grid[0].legend(loc = (0, -.33), ncol = 5, fontsize = 12)
        for i in range(len(self.data.Sub_Zone.unique())):
            lgnd.legendHandles[i]._sizes = [40]

        # density map colourbar 
        cbar = fig.colorbar(heatmap, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Magnitude $\geq$ 8.5 density difference', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def pc_magnitude_plots(self):
        ''' 
        Creates plots of the PC projections colour coded by maximum magnitude. 
        
        Called when this class is initiated. 
        '''
        
        # define figure, subplots, and colourbar
        fig = plt.figure(figsize = (15,4))
        grid = ImageGrid(fig, 111,
                        nrows_ncols = (1,3),
                        axes_pad = 1,
                        label_mode="all",
                        cbar_location = "right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.3, 
                        aspect=False)

        # define discrete colourbar
        cmap = (mpl.colors.ListedColormap(['tab:green', 'tab:blue', 'tab:orange']))
        bounds = [0, 4, 8.5, 10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        # sort data by maximum magnitude to prepare for plotting 
        high_mag = self.data[self.data.Max_mag >= 8.5]
        mid_mag = self.data[(self.data.Max_mag >= 4) & (self.data.Max_mag < 8.5)]
        low_mag = self.data[self.data.Max_mag < 4]
        
        # 3 scatterplots: PC1 vs. PC2, PC2 vs. PC3, PC3 vs. PC4 
        for i in range(3):
            for subset in [mid_mag, low_mag, high_mag]:
                plot = grid[i].scatter(subset[f'PC{i+1}'], subset[f'PC{i+2}'], c = subset.Max_mag, cmap=cmap, norm=norm, alpha = 0.7)

        # set axes labels and plot titles
        for i in range(3): 
            grid[i].set_xlabel(f'PC{i+1}')
            grid[i].set_ylabel(f'PC{i+2}')
            grid[i].set_title(f'PC{i+1} vs. PC{i+2} by maximum magnitude')
        
        # set colorbar
        cbar = fig.colorbar(plot, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Maximum magnitude', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def feature_distribution_plots(self):
        ''' 
        Creates plots of the PC projections colour coded by each feature (margin property).
        
        Called when this class is initiated. 
        
        Calls: 
        - get_feature_dicts
        - mag_range_density
        - feature_plots
        '''
        
        # define figure and subplots
        fig,ax = plt.subplots(2,3, figsize = (15,8))
        fig.tight_layout(pad = 4)
        fig.delaxes(ax[1,2])

        # create a feature PC space distribution plot for each feature:
        axes = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1]]
        feature_dict, unit_dict = self.get_feature_dicts()

        
        for i in range(5): 
            # scatter data colour coded by feature:             
            vmin, vmax = self.data[self.features[i]].min(), np.percentile(self.data[self.features[i]], 90)
            feature_plot = axes[i].scatter(self.data['PC1'], self.data['PC2'], s = 40, c = self.data[self.features[i]], \
                             cmap = 'coolwarm', alpha = .5,  vmin=vmin, vmax=vmax)
            
            # density contours of points with maximum magnitude >= 8.5, and of those with 7 =< maximum magnitude < 8.5
            self.mag_range_density(ax=axes[i], mag_range=[8.5, 11], linestyle='solid')
            self.mag_range_density(ax=axes[i], mag_range=[7, 8.5], linestyle='dashed')
            
            # set plot title, axis limits and labels 
            axes[i].set_title(feature_dict[self.features[i]], size = 14)
            axes[i].set_xlim([self.xmin, self.xmax])
            axes[i].set_ylim([self.ymin, self.ymax])
            axes[i].set_xlabel('PC1')
            axes[i].set_ylabel('PC2')
            
            #set colourbar
            cbar = fig.colorbar(feature_plot, ax=axes[i], extend = 'both')
            cbar.set_label(label=f'{feature_dict[self.features[i]]} ({unit_dict[self.features[i]]})', fontsize = 12)
            cbar.ax.tick_params(labelsize=12)

        # create a legend to label the density contours 
        
        txt = '$M_{max}$'
        handles = [mpatches.Patch(edgecolor='k', facecolor='none', label=f'Density of segments with {txt} $\geq$ 8.5'),\
                  mpatches.Patch(edgecolor='k', facecolor='none', linestyle='--', \
                                 label = f'Density of segments with 7 $\leq$ {txt} < 8.5')]
        fig.legend(handles=handles, ncol = 2, bbox_to_anchor=(.75, 0.03), fontsize = 12) 
        
        
    def pc_connections_plots(self):
        ''' 
        Creates connection plots of the principal components colour coded by maximum magnitude. 
        
        Called when this class is initiated. 
        '''
        
        # prepare data for plotting
        plotdata = self.data.copy()
        labels = ['$M_{max} \geq$ 8.5', '4 $ \leq M_{max}$ < 8.5', '$M_{max}$ < 4']
        plotdata['label'] = [labels[0] if mag >= 8.5 else labels[1] if mag >=4 else labels[2] for mag in self.data.Max_mag]
        plotdata = plotdata.drop(columns = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough', 'Max_mag', 'Sub_Zone', 'Longitude', 'Latitude'])

        # define figure and subplots, set title and y-axis label 
        fig,ax = plt.subplots(1,3, figsize = (15,5))
        fig.suptitle('Principal components by maximum magnitude')
        ax[0].set_ylabel('PC values')
        
        # subplot 0: all data
        parallel_coordinates(plotdata[plotdata.label == labels[1]], 'label', color='tab:blue', alpha=.5, ax=ax[0])
        parallel_coordinates(plotdata[plotdata.label == labels[0]], 'label', color='tab:orange', alpha=.5, ax=ax[0])
        parallel_coordinates(plotdata[plotdata.label == labels[2]], 'label', color='tab:green', alpha=.5, ax=ax[0])
        
        # subplot 1: only maximum magnitudes < 4 or >= 8.5
        parallel_coordinates(plotdata[plotdata.label == labels[0]], 'label', color='tab:orange', alpha=.5, ax=ax[1])
        parallel_coordinates(plotdata[plotdata.label == labels[2]], 'label', color='tab:green', alpha=.5, ax=ax[1])
        
        # subplot 2: only maximum magnitudes between 4 and 8.5
        parallel_coordinates(plotdata[plotdata.label == labels[1]], 'label', color='tab:blue', alpha=.5, ax=ax[2])
     
    
    def heatmap_diff(self, ax, threshold = 8.5):
        '''
        Calculates a density difference map of segments with maximum magnitude above the threshold and overall segment distribution
        in the PC space. 
        
        Called in: 
        - pc_space_plots
        
        Parameters: 
        - ax: axis at which the density map should be plotted
        - threshold: float, earthquake magnitude value above which the density of high-magnitude segments is calculated, default = 8.5
        
        Returns:
        - heatmap: the density difference map 
        '''

        # calculate overall density in the PC space
        x1, y1 = np.array(self.data.PC1, dtype = float), np.array(self.data.PC2, dtype = float)
        xi, yi = np.mgrid[self.xmin:self.xmax:x1.size**0.5*1j, self.ymin:self.ymax:y1.size**0.5*1j]
        k1 = gaussian_kde(np.vstack([x1, y1]))
        zi1 = k1(np.vstack([xi.flatten(), yi.flatten()]))

        # calculate density of maximum magnitudes above threshold in the PC space
        heatmap_data = self.data[self.data.Max_mag >= threshold]
        x2, y2 = np.array(heatmap_data.PC1, dtype = float), np.array(heatmap_data.PC2, dtype = float)
        k2 = gaussian_kde(np.vstack([x2, y2]))
        zi2 = k2(np.vstack([xi.flatten(), yi.flatten()]))

        # calculate the difference density map
        zi = zi2-zi1 # density of maximum magnitudes above threshold - overall density
        
        # plot density map
        heatmap = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap = 'coolwarm', alpha = 0.7)

        return heatmap
    
        
    def pc_space_distribution(self, ax, data):
        '''
        Scatters PC projection data colour coded by subduction margins. 
        
        Called in: 
        - pc_space_plots
        
        Calls: 
        - get_zone_dicts
        
        Parameters: 
        - ax: axis at which the PC space distribution of segments should be plotted
        - data: dataframe containing segments to be plotted here 
        '''
    
        zones, zone_color_dict, zone_label_dict = self.get_zone_dicts()

        for zone in zones:
            if zone in data.Sub_Zone.unique():
                zonedata = data[data.Sub_Zone == zone]
                ax.scatter(zonedata.PC1, zonedata.PC2, c = zone_color_dict[zone], s = 10, label = zone_label_dict[zone], alpha = .5)

            else: 
                pass
            
            
    def mag_range_density(self, ax, mag_range, linestyle='solid'):
        '''
        Calculates and plots density contours of segments with maximum magnitude within the specified range.
        
        Called in: 
        -pc_magnitude_plots
        
        Parameters: 
        - ax: axis at which the density contours should be plotted
        - mag_range: list containing two values describing the magnitude range of the density contours to be plotted
        - linestyle: linestyle of the density contours, e.g. 'dashed'. Default is 'solid'
        '''
        
        # select data within magnitude range
        mag_range_data = self.data[(self.data['Max_mag'] >= mag_range[0]) & (self.data['Max_mag'] < mag_range[1])]

        # Peform the kernel density estimate
        xx, yy = np.mgrid[self.xmin:self.xmax:100j, self.ymin:self.ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([mag_range_data.PC1, mag_range_data.PC2])
        kernel = gaussian_kde(np.array(values, dtype = 'float64'))
        f = np.reshape(kernel(positions).T, xx.shape)

        # plot density contours
        cset = ax.contour(xx, yy, f, linestyles = linestyle, colors='k', levels = [.3,.45,.6])
        ax.clabel(cset, inline=1, fontsize=10) # to display the contour levels 
        
            
    def get_low_margin_data(self, threshold = 8.5):
        '''
        Filters data for margins with maximum magnitudes below a threshold (default M8.5).
        
        Called in: 
        - pc_space_plots
        
        Parameters: 
        - threshold: float, earthquake magnitude value below which margins should be inlcuded in the returned dataframe
        
        Returns: 
        - a dataframe containing all segments from margins where the highest magnitude is below the threshold
        '''
        
        below = []

        for zone in self.data.Sub_Zone.unique():
            zonedata = self.data[self.data.Sub_Zone == zone]
            if max(zonedata.Max_mag) < threshold:
                below.append(zone)

        return self.data[self.data.Sub_Zone.isin(below)]
        
        
    def get_feature_dicts(self):
        '''
        Defines and returns dictionaries containing full names for the features, and their units
        
        Called in: 
        - feature_distribution_plots
        
        Returns: 
        - feature_dict: dictionary containing the abbreviated feature names and their full names
        - unit_dict: dictionary containing the abbreviated feature names and their units
        '''

        feature_dict = {'Sed_Thick': 'Sediment thickness', 'Age': 'Plate Age', 'Dip': 'Dip angle',\
                    'Vel': 'Relative plate velocity', 'Rough': 'Roughness'}

        unit_dict = {'Sed_Thick': 'm', 'Age': 'Ma', 'Dip': '°', 'Vel': 'mm/yr', 'Rough': 'mGal'}
        return feature_dict, unit_dict
    
    
    def get_zone_dicts(self):
        '''
        Defines and returns a list of all considered subduction margins as well as dictionaries containing the margin's full 
        names and their assigned colours for plotting
        
        Called in: 
        - pc_space_distribution
        
        Returns: 
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

        return zones, zone_color_dict, zone_label_dict
    
  
                
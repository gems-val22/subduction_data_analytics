import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde
from pandas.plotting import parallel_coordinates

import modules.graphics as graph


class FigureMaker():
    
    '''
    Class to create figures to show the PC projections. 
    
    Called in the class Projector (if the input parameter generate_figures is set to True).
    '''
    
    def __init__(self, data):
        
        self.data = data
        self.xmin, self.xmax, self.ymin, self.ymax = data.PC1.min()*1.25, data.PC1.max()*1.25, data.PC2.min()*1.25, data.PC2.max()*1.25
        
        self.features = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough']
        self.feature_dict, self.unit_dict = graph.get_feature_dicts()
        
        self.pc_space_plots()
        self.feature_distribution_plots()
        self.pc_magnitude_plots()
        self.pc_connections_plots()
        
        
    def pc_space_plots(self):
        ''' 
        Function to create plots of the PC projections colour coded by each feature (margin property).
        
        Called when this class is initiated. 
        
        Calls:
        - pc_space_distribution from graphics.py
        - heatmap_diff from graphics.py
        - get_low_margin_data from graphics.py
        - 
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
                        cbar_pad=0.3
                        )

        # subplot 0: scatter plot the PC space distribution colour coded by margins
        graph.pc_space_distribution(grid[0], self.data)
        
        # subplot 1: PC space density map for M >= 8.5 segments and PC space distribution of margins with maximum magnitude < 8.5
        heatmap = graph.heatmap_diff(fig, grid[1], self.data, cbar=False)
        graph.pc_space_distribution(grid[1], graph.get_low_margin_data(self.data), legend = False, axes_adjust=False)

        # set axes titles and density map colourbar 
        grid[0].set_title('PC space by margin', fontsize = 14)
        grid[1].set_title('Hazard assessment', fontsize = 14)
        cbar = fig.colorbar(heatmap, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Magnitude $\geq$ 8.5 density difference', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def feature_distribution_plots(self):
        ''' 
        Function to create plots of the PC projections colour coded by each feature (margin property).
        
        Called when this class is initiated. 
        
        Calls: 
        - mag_range_density()
        - feature_plots
        '''
        
        # define figure and subplots
        fig,ax = plt.subplots(2,3, figsize = (15,8))
        fig.tight_layout(pad = 4)
        fig.delaxes(ax[1,2])

        # create a feature PC space distribution plot for each feature:
        axes = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1]]
        
        for i in range(5): 
            # scatter data colour coded by feature: 
            feature_plot = self.feature_plots(self.features[i], axes[i]) 
            
            # density contours of points with maximum magnitude >= 8.5, and of those with 7 =< maximum magnitude < 8.5
            self.mag_range_density(axes[i], [8.5, 11], 'solid')
            self.mag_range_density(axes[i], [7, 8.5], 'dashed')
            
            # set axis limits and labels 
            axes[i].set_xlim([self.xmin, self.xmax])
            axes[i].set_ylim([self.ymin, self.ymax])
            axes[i].set_xlabel('PC1')
            axes[i].set_ylabel('PC2')
            
            #set colourbar
            cbar = fig.colorbar(feature_plot, ax=axes[i], extend = 'both')
            cbar.set_label(label=f'{self.feature_dict[self.features[i]]} ({self.unit_dict[self.features[i]]})', fontsize = 12)
            cbar.ax.tick_params(labelsize=12)

        # create a legend to label the density contours 
        
        txt = '$M_{max}$'
        handles = [mpatches.Patch(edgecolor='k', facecolor='none', label=f'Density of segments with {txt} $\geq$ 8.5'),\
                  mpatches.Patch(edgecolor='k', facecolor='none', linestyle='--', \
                                 label = f'Density of segments with 7 $\leq$ {txt} < 8.5')]
        fig.legend(handles=handles, ncol = 2, bbox_to_anchor=(.75, 0.03), fontsize = 12) 
        
        
    def pc_magnitude_plots(self):
        ''' 
        Function to create plots of the PC projections colour coded by maximum magnitude. 
        
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
                        aspect=False
                        )

        # define discrete colourbar
        cmap = mpl.cm.plasma
        cmap = (mpl.colors.ListedColormap(['k','tab:purple', 'tab:blue', 'tab:orange', 'tab:red'])
            .with_extremes(under='tab:purple', over='tab:orange'))
        bounds = [0, 4, 8.5, 10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        
        # sort data by maximum magnitude to prepare for plotting
        plotdata = self.data.sort_values(by = 'Max_mag', ascending = True) 

        # 3 scatterplots: PC1 vs. PC2, PC2 vs. PC3, PC3 vs. PC4 
        plotdata_ = grid[0].scatter(plotdata['PC1'], plotdata['PC2'], c = plotdata.Max_mag, cmap = cmap, norm=norm, alpha = .5)
        plotdata_ = grid[1].scatter(plotdata['PC2'], plotdata['PC3'], c = plotdata.Max_mag, cmap = cmap, norm=norm, alpha = .5)
        plotdata_ = grid[2].scatter(plotdata['PC3'], plotdata['PC4'], c = plotdata.Max_mag, cmap = cmap, norm=norm, alpha = .5)

        # set axes labels and plot titles
        for i in range(3): 
            grid[i].set_xlabel(f'PC{i+1}')
            grid[i].set_ylabel(f'PC{i+2}')
            grid[i].set_title(f'PC{i+1} vs. PC{i+2} by maximum magnitude')
        
        # set colorbar
        cbar = fig.colorbar(plotdata_, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Maximum magnitude', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def pc_connections_plots(self):
        ''' 
        Function to create connection plots of the principal components colour coded by maximum magnitude. 
        
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
        parallel_coordinates(plotdata[plotdata.label == labels[2]], 'label', color='tab:purple', alpha=.5, ax=ax[0])
        
        # subplot 1: only maximum magnitudes < 4 or >= 8.5
        parallel_coordinates(plotdata[plotdata.label == labels[0]], 'label', color='tab:orange', alpha=.5, ax=ax[1])
        parallel_coordinates(plotdata[plotdata.label == labels[2]], 'label', color='tab:purple', alpha=.5, ax=ax[1])
        
        # subplot 2: only maximum magnitudes between 4 and 8.5
        parallel_coordinates(plotdata[plotdata.label == labels[1]], 'label', color='tab:blue', alpha=.5, ax=ax[2])
                
            
    def mag_range_density(self, ax, mag_range, linestyle):
        
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
        
        
    def feature_plots(self, feature, ax):
        '''
        Called in: 
        - feature_distribution_plots
        
        Calls: 
        - get_feature_dicts()
        '''
        
        vmin, vmax = self.data[feature].min(), np.percentile(self.data[feature], 90)

        ax.scatter(self.data['PC1'], self.data['PC2'], s = 60, c = 'white')
        ft_plot = ax.scatter(self.data['PC1'], self.data['PC2'], s = 40, c = self.data[feature], \
                             cmap = 'coolwarm', alpha = .5,  vmin=vmin, vmax=vmax)

        ax.set_xlim([self.xmin, self.xmax])
        ax.set_ylim([self.ymin, self.ymax])

        feature_dict, unit_dict = self.get_feature_dicts()
        ax.set_title(feature_dict[feature], size = 14)

        return ft_plot 
        
        
    def get_feature_dicts(self):

        feature_dict = {'Sed_Thick': 'Sediment thickness', 'Age': 'Plate Age', 'Dip': 'Dip angle',\
                    'Vel': 'Relative plate velocity', 'Rough': 'Roughness'}

        unit_dict = {'Sed_Thick': 'm', 'Age': 'Ma', 'Dip': '°', 'Vel': 'mm/yr', 'Rough': 'mGal'}
        return feature_dict, unit_dict


                
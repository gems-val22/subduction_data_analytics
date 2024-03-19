import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
from pandas.plotting import parallel_coordinates

import modules.graphics as graph


class FigureMaker():
    
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

        graph.pc_space_distribution(grid[0], self.data)
        heatmap = graph.heatmap_diff(fig, grid[1], self.data, cbar=False)
        graph.pc_space_distribution(grid[1], graph.get_low_margin_data(self.data), legend = False, axes_adjust=False)


        grid[0].set_title('PC space by margin', fontsize = 14)
        grid[1].set_title('Hazard assessment', fontsize = 14)

        cbar = fig.colorbar(heatmap, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Magnitude $\geq$ 8.5 density difference', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def feature_distribution_plots(self):

        fig,ax = plt.subplots(2,3, figsize = (15,8))
        fig.tight_layout(pad = 4)
        fig.delaxes(ax[1,2])

        axes = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1]]

        for i in range(5): 
            feature_plot = graph.plotting_kriging_map(self.data, self.features[i], axes[i]) 
            txt, threshold = graph.mag_density(self.data, axes[i])
            txt, mag_range = graph.mag_range_density(self.data, axes[i])
            graph.pc_axis_labels(axes[i], self.xmin, self.xmax, self.ymin, self.ymax)

            cbar = fig.colorbar(feature_plot, ax=axes[i], extend = 'both')
            cbar.set_label(label=f'{self.feature_dict[self.features[i]]} ({self.unit_dict[self.features[i]]})', fontsize = 12)
            cbar.ax.tick_params(labelsize=12)

        handles = [mpatches.Patch(edgecolor='k', facecolor='none', label=f'Density of segments with {txt} $\geq$ {threshold}'),\
                  mpatches.Patch(edgecolor='k', facecolor='none', linestyle='--', \
                                 label = f'Density of segments with {mag_range[0]} $\leq$ {txt} < {mag_range[1]}')]

        fig.legend(handles=handles, ncol = 2, bbox_to_anchor=(.75, 0.03), fontsize = 12) 
        
        
    def pc_magnitude_plots(self):
        
        fig = plt.figure(figsize = (15,4))
        grid = ImageGrid(fig, 111,
                        nrows_ncols = (1,3),
                        axes_pad = .7,
                        label_mode="all",
                        cbar_location = "right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.3, 
                        aspect=False
                        )

        cmap = mpl.cm.plasma
        cmap = (mpl.colors.ListedColormap(['k','tab:purple', 'tab:blue', 'tab:orange', 'tab:red'])
            .with_extremes(under='tab:purple', over='tab:orange'))
        bounds = [0, 4, 8.5, 10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

        plotdata = self.data.sort_values(by = 'Max_mag', ascending = True)

        pcs = ['PC1', 'PC2', 'PC3', 'PC4']

        for i in range(3):
            plotdata_ = grid[i].scatter(plotdata[pcs[i]], plotdata[pcs[i+1]], c = plotdata.Max_mag, cmap = cmap, norm=norm, alpha = .5)
            grid[i].set_xlabel(pcs[i])
            grid[i].set_ylabel(pcs[i+1])
            grid[i].set_title(f'{pcs[i]} vs. {pcs[i+1]} by maximum magnitude')

        cbar = fig.colorbar(plotdata_, cax=grid.cbar_axes[0])
        cbar.set_label(label=f'Maximum magnitude', fontsize = 12)
        cbar.ax.tick_params(labelsize=12)
        
        
    def pc_connections_plots(self):
        
        plotdata = self.data.copy()
        plotdata.sort_values(by = 'Max_mag', ascending = True, inplace = True)
        
        labels = ['$M_{max} \geq$ 8.5', '4 $ \leq M_{max}$ < 8.5', '$M_{max}$ < 4']
        plotdata['label'] = [labels[0] if mag >= 8.5 else labels[1] if mag >=4 else labels[2] for mag in self.data.Max_mag]
        
        plotdata = plotdata.drop(columns = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough', 'Max_mag', 'Sub_Zone', 'Longitude', 'Latitude'])

        fig,ax = plt.subplots(1,3, figsize = (15,5))
        fig.suptitle('Principal components by maximum magnitude')

        for i in range(3):
            if i == 0 or i == 2:
                parallel_coordinates(plotdata[plotdata.label == labels[1]], 'label', color='tab:blue', alpha=.5, ax=ax[i])
            if i == 0 or i == 1:
                parallel_coordinates(plotdata[plotdata.label == labels[0]], 'label', color='tab:orange', alpha=.5, ax=ax[i])
                parallel_coordinates(plotdata[plotdata.label == labels[2]], 'label', color='tab:purple', alpha=.5, ax=ax[i])
                
                
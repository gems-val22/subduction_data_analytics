import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from subductionPCA.FigureMaker import FigureMaker


class Projector():
    """
    A class for projecting and visualizing data using dimensionality reduction techniques.

    Parameters:
    -----------
    params : dict
        Dictionary containing parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

    features : list, optional, default: ['Sed_Thick', 'Dip', 'Vel', 'Rough']
        List of feature names.

    generate_figures : bool, optional, default: False
        Whether to generate figures after projection.

    Attributes:
    -----------
    features : list
        List of features to be considered for PCA.

    n_components : int
        Number of components for PCA.

    all_features : list
        List of all feature names.

    threshold : int
        Threshold maximum magnitude value.

    data : pd.DataFrame
        Loaded and preprocessed subduction margin property dada.

    params : dict
        Parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

    scaler : object
        Scaler object (scikit-learn's StandardScaler, RobustScaler, or MinMaxScaler).

    pca : object
        PCA object (PCA or Kernel-PCA). 

    projector : object
        Projector object (containing data preprocessing and (Kernel)-PCA projection). 

    data_projected : pd.DataFrame
        PCA-projected data.
        
    cumulative_explained_variance_ratio : np.array
        Cumulative variance explained for the PCs.
        
    ari : float
        Adjusted Rand Index score for this PCA projection. 
    """

    
    def __init__(self, params, features=['Sed_Thick', 'Dip', 'Vel', 'Rough'], generate_figures=False):
        """
        Initializes the Projector.

        Parameters:
        -----------
        params : dict
            Dictionary containing parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

        features : list, optional, default: ['Sed_Thick', 'Dip', 'Vel', 'Rough']
            List of feature names.

        generate_figures : bool, optional, default: False
            Whether to generate figures after projection.
        """
        
        self.features = features
        self.n_components = len(self.features)
        self.all_features = ['Sed_Thick', 'Dip', 'Vel', 'Rough']
        
        self.threshold = 8.5 # figures with different thresholds are generated (if generate_figures is True) 

        # data loading and preprocessing
        self.data = self.load_data()

        # projector parameters
        self.params = params
        self.scaler = self.get_scaler()
        self.pca = self.define_pca()

        # PCA projections
        self.projector = self.get_projector()
        self.data_projected = self.project(self.data)
        
        # calculate explained variance
        self.cumulative_explained_variance_ratio = self.calculate_variance_explained() 
        
        # get ARI score
        self.ari = self.calculate_ari_score() 

        # generate figures
        if generate_figures:
            FigureMaker(projector = self)
            
        # for linear PCA: display components    
        if self.params['kernel'] is None:
            W = pd.DataFrame(self.pca.components_.T, index=self.features, columns=[f'PC{i}' for i in range(1, len(self.features)+1)])
            display(W)
            
            
    def load_data(self, path='data/preprocessed_data_ghea.csv'):
        """
        Loads the data and drop unnecessary columns.

        Parameters:
        -----------
        path : str, optional, default: 'data/preprocessed_data_ghea.csv'
            File path to the preprocessed data.

        Returns:
        --------
        all_data : pd.DataFrame
            DataFrame containing subduction margin properties, assigned maximum magnitudes, segment locations, and margin names.
        """
        
        all_data = pd.read_csv(path).drop(columns=['Unnamed: 0', 'Segment', 'sSSE', 'lSSE'])
        return all_data

    
    def get_preprocessors(self):
        """
        Creates and returns the preprocessing pipelines.

        Returns:
        --------
            pca_preproc: ColumnTransformer object
                Includes a log transform for Sediment thickness, dip, roughness, and iputing and scaling for all features.
                This object is used to prepare features for applying PCA. All other columns are dropped. 
            feature_preproc: ColumnTransformer object
                Includes an imputer for the features and keeps all other columns. 
        """
        
        to_log = [ft for ft in self.features if ft in ['Sed_Thick', 'Dip', 'Rough']]

        log_preproc = Pipeline([('Imputer', SimpleImputer(strategy='mean')),
                        ('Log Transform', LogTransformer()),
                        ('Scaler', self.scaler)])

        no_log_preproc = Pipeline([('Imputer', SimpleImputer(strategy='mean')),
                                    ('Scaler', self.scaler)])

        pca_preproc = ColumnTransformer([
            ('Log. Preprocessor', log_preproc, to_log),
            ('Preprocessor', no_log_preproc, [ft for ft in self.features if ft not in to_log])],
            remainder='drop', force_int_remainder_cols=False)
        
        feature_preproc = ColumnTransformer([
            ('Imputer', SimpleImputer(strategy='mean'), self.all_features)],
            remainder='passthrough', force_int_remainder_cols=False)
        
        return pca_preproc, feature_preproc
    
    
    def get_projector(self):
        """
        Creates and returns the projector object.

        Returns:
        --------
        pca_projector : FeatureUnion object 
            Projector object (containing data preprocessing and (Kernel)-PCA projection). 
        """

        pca_preproc, feature_preproc = self.get_preprocessors() 

        pca_pipe = Pipeline([
            ('preprocessing', pca_preproc),
            ('PCA', self.pca)])
        
        pca_projector = FeatureUnion([('Feature processing', feature_preproc),
                                      ('PCA projection', pca_pipe)])

        pca_projector.fit(self.data)

        return pca_projector

    
    def project(self, X):
        """
        PCA-projects the data.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data (margin property data).

        Returns:
        --------
        X_projected : pd.DataFrame
            PCA-projected data.
        """
        
        new_columns = self.all_features + [col for col in X.columns if col not in self.all_features] + \
                        [f'PC{i+1}' for i in range(self.n_components)]

        X_projected = pd.DataFrame(self.projector.transform(X), columns=new_columns)

        return X_projected
    
    
    def calculate_variance_explained(self):
        '''
        Calculates the cululative explained variance ratio for all PCs. 
        
        Returns:
        --------
        cumulative_explained_variance_ratio : np.array 
            Cumulative explained variance ratio for each PC. 
        '''

        pca_preproc, _ = self.get_preprocessors() 
    
        pca_pipe = Pipeline([
            ('preprocessing', pca_preproc),
            ('PCA', self.define_pca(limit_components = False))])
        
        X_proj = pca_pipe.fit_transform(self.data)
        X_proj = pd.DataFrame(X_proj, columns = [f'PC{i}' for i in range(1,X_proj.shape[1]+1)]) 
        
        explained_variance_ratio = (X_proj.std()**2 / ((X_proj.std()**2).sum())).values
        cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
                    
        return cumulative_explained_variance_ratio
    
    
    def calculate_ari_score(self):
        '''
        Calculates the Adjusted Rand Index score for the PCA projection. This is 1 for PCA and <1 for Kernel-PCA. 
        
        Returns:
        --------
        ari : float
            Adjusted Rand Index score for the PCA projection. 
        '''
        
        if self.params['kernel'] is None:
            return 1
        
        else:                        
            pca_preproc, _ = self.get_preprocessors() 
    
            pca_pipe = Pipeline([
                    ('preprocessing', pca_preproc),
                    ('PCA', PCA(n_components=self.n_components, random_state=42))])
                    
            X_proj = pca_pipe.fit_transform(self.data)
            X_proj = pd.DataFrame(X_proj, columns = [f'PC{i}' for i in range(1,X_proj.shape[1]+1)])    
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            pca_clusters = kmeans.fit_predict(X_proj[['PC1', 'PC2']])
            kpca_clusters = kmeans.fit_predict(self.data_projected[['PC1', 'PC2']])
            
            ari = adjusted_rand_score(pca_clusters, kpca_clusters)
            
            return ari
    
    
    def get_scaler(self):
        """
        Creates the scaler object based on input parameters.

        Returns:
        --------
        scaler : object
            Scaler object.
        """
        
        if self.params['scaler'] == 'StandardScaler':
            scaler = StandardScaler()
        elif self.params['scaler'] == 'RobustScaler':
            scaler = RobustScaler()
        elif self.params['scaler'] == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError('Scaler input not valid')

        return scaler

 
    def define_pca(self, limit_components = True):
        """
        Defines the PCA object based on input parameters.

        Returns:
        --------
        pca : object
            PCA object.
        """
            
        if self.params['kernel'] is None and limit_components == True:
            pca = PCA(n_components=self.n_components, random_state=42)
        elif self.params['kernel'] is None and limit_components == False: 
            pca = PCA(random_state=42)
        elif self.params['kernel'] is not None and limit_components == True:
            pca = KernelPCA(kernel=self.params['kernel'], n_components=self.n_components, random_state=42)
        else:
            pca = KernelPCA(kernel=self.params['kernel'], random_state=42)

        return pca



class LogTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that applies natural logarithm transformation to the input data.

    Attributes:
    -----------
    seed : float
        The seed value used for adding to the input data before logarithmic transformation.
    """

    def __init__(self):
        """
        Initialize the LogTransformer.

        Parameters:
        -----------
        seed : float, optional, default: 1e-5
            A small value added to the input data before applying logarithm to avoid taking the logarithm of zero or negative values.
        """
        self.seed = 1e-5

        
    def fit(self, X):
        """
        Fit the LogTransformer to the data. This method does nothing as this transformer does not learn anything from the data.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        return self

    
    def transform(self, X, y=None):
        """
        Transform the input data by applying natural logarithm.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        X_transformed : array-like or sparse matrix, shape (n_samples, n_features)
            The transformed data after applying natural logarithm to each element.
        """
        return np.log(X + self.seed)
    
    
    
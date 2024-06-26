import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler 
from sklearn.decomposition import PCA, KernelPCA

from subductionPCA.FigureMaker import FigureMaker


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


class Projector():
    """
    A class for projecting and visualizing data using dimensionality reduction techniques.

    Parameters:
    -----------
    params : dict
        Dictionary containing parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

    features : list, optional, default: ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough']
        List of feature names.

    generate_figures : bool, optional, default: False
        Whether to generate figures after projection.

    Attributes:
    -----------
    features : list
        List of feature names.

    n_components : int
        Number of components for PCA.

    all_features : list
        List of all feature names.

    threshold : int
        Threshold value.

    data : DataFrame
        Loaded and preprocessed subduction margin property dada.

    params : dict
        Parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

    scaler : object
        Scaler object (scikit-learn's StandardScaler, RobustScaler, or MinMaxScaler).

    pca : object
        PCA object (PCA or Kernel-PCA). 

    projector : object
        Projector object (containing data preprocessing and (Kernel)-PCA projection). 

    data_projected : DataFrame
        PCA-projected data.
    """

    
    def __init__(self, params, features=['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough'], generate_figures=False):
        """
        Initialize the Projector.

        Parameters:
        -----------
        params : dict
            Dictionary containing parameters for scaler (scaler type) and PCA (kernel type, or no kernel (None)).

        features : list, optional, default: ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough']
            List of feature names.

        generate_figures : bool, optional, default: False
            Whether to generate figures after projection.
        """
        self.features = features
        self.n_components = 5
        self.all_features = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough']
        self.threshold = 8

        # data loading and preprocessing
        self.data = self.load_data()

        # projector parameters
        self.params = params
        self.scaler = self.get_scaler()
        self.pca = self.define_pca()

        # PCA projections
        self.projector = self.get_projector()
        self.data_projected = self.project(self.data)

        # generate figures
        if generate_figures:
            FigureMaker(self.data_projected)

            
    def load_data(self, path='data/preprocessed_data.csv'):
        """
        Load the data and drop unnecessary columns.

        Parameters:
        -----------
        path : str, optional, default: 'data/preprocessed_data.csv'
            File path to the preprocessed data.

        Returns:
        --------
        all_data : DataFrame
            Loaded and preprocessed subduction margin property dada.
        """
        
        all_data = pd.read_csv(path).drop(columns=['Unnamed: 0', 'Segment', 'sSSE', 'lSSE'])
        return all_data

    
    def get_projector(self):
        """
        Create and return the projector object.

        Returns:
        --------
        pca_projector : object
            Projector object (containing data preprocessing and (Kernel)-PCA projection).
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
            remainder='drop')

        pca_pipe = Pipeline([
            ('preprocessing', pca_preproc),
            ('PCA', self.pca)])

        feature_preproc = ColumnTransformer([
            ('Imputer', SimpleImputer(strategy='mean'), self.all_features)],
            remainder='passthrough')

        pca_projector = FeatureUnion([('Feature processing', feature_preproc),
                                      ('PCA projection', pca_pipe)])

        pca_projector.fit(self.data)

        return pca_projector

    
    def project(self, X):
        """
        PCA-project the data.

        Parameters:
        -----------
        X : DataFrame
            Input data (margin property data).

        Returns:
        --------
        X_projected : DataFrame
            PCA-projected data.
        """
        new_columns = self.all_features + [col for col in X.columns if col not in self.all_features] + \
                        [f'PC{i+1}' for i in range(self.n_components)]

        X_projected = pd.DataFrame(self.projector.transform(X), columns=new_columns)

        return X_projected

    
    def get_scaler(self):
        """
        Get the scaler object based on input parameters.

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

    
    def define_pca(self):
        """
        Define the PCA object based on input parameters.

        Returns:
        --------
        pca : object
            PCA object.
        """
        if self.params['kernel'] is None:
            pca = PCA(n_components=self.n_components, random_state=42)
        else:
            pca = KernelPCA(kernel=self.params['kernel'], n_components=self.n_components, random_state=42)

        return pca

    

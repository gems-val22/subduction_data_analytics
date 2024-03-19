import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler 
from sklearn.decomposition import PCA, KernelPCA

from FigureMaker import FigureMaker


class LogTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, seed = 1e-5):
        self.seed=seed
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.log(X+self.seed)

    
class Projector():
    
    def __init__(self, params, features = ['Sed_Thick', 'Age', 'Dip', 'Vel', 'Rough'], generate_figures = False):
        
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
        if generate_figures == True: 
            FigureMaker(self.data_projected)

        
    def load_data(self, path = 'data/preprocessed_data.csv'):
        
        all_data = pd.read_csv(path).drop(columns = ['Unnamed: 0', 'Segment', 'sSSE', 'lSSE'])
        
        return all_data
    
    
    def get_projector(self): 
        
        to_log = [ft for ft in self.features if ft in ['Sed_Thick', 'Dip', 'Rough']]
        
        log_preproc =  Pipeline([('Imputer', SimpleImputer(strategy = 'mean')), 
                                    ('Log Transform', LogTransformer()),
                                    ('Scaler', self.scaler)])

        no_log_preproc = Pipeline([('Imputer', SimpleImputer(strategy = 'mean')), 
                                    ('Scaler', self.scaler)])

        pca_preproc = ColumnTransformer([
            ('Log. Preprocessor', log_preproc, to_log),
            ('Preprocessor', no_log_preproc, [ft for ft in self.features if ft not in to_log])], 
            remainder = 'drop')

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
                
        new_columns = self.all_features + [col for col in X.columns if col not in self.all_features] + \
                        [f'PC{i+1}' for i in range(self.n_components)]
                                                   
        X_projected = pd.DataFrame(self.projector.transform(X), columns = new_columns)
               
        return X_projected
    
    
    def get_scaler(self):

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

        if self.params['kernel'] == None: 
            pca = PCA(n_components=self.n_components, random_state=42)
        else: 
            pca = KernelPCA(kernel=self.params['kernel'], n_components=self.n_components, random_state=42)

        return pca
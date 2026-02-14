"""
Modelos supervisados del proyecto.

- LGBMModel: Gradient Boosting con LightGBM.
"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from imblearn.pipeline import Pipeline, make_pipeline
from src.preprocessing.preprocessing  import ToDummy, TeEncoder, CardinalityReducer, MinMaxScalerRow
from lightgbm import LGBMClassifier, early_stopping
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer


def binary_area_peligrosa(x):
    return x.map({'zona peligrosa':1,'zona no peligrosa':0}).to_frame()

def get_preprocesor(preprocesor):
    """Preprocesador para features categóricas. Soporta: -1 (passthrough), 1 (dummy + TE estrato/localidad)."""
    if preprocesor == -1:
        preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')
    elif preprocesor == 1:
        pipe_estrato = Pipeline([
            ('cardinality_reducer', CardinalityReducer(threshold=0.05)),
            ('te', TeEncoder(['subcategoria_estrato'], w=50))
        ])
        pipe_localidad = Pipeline([
            ('cardinality_reducer', CardinalityReducer(threshold=0.005)),
            ('te', TeEncoder(['localidad'], w=50))
        ])
        vars_dummy = ['categoria']
        t_features = [
            ('dummy_var', ToDummy(vars_dummy), vars_dummy),
            ('p_localidad', pipe_localidad, ['localidad']),
            ('p_estrato', pipe_estrato, ['subcategoria_estrato']),
        ]
        preprocessor = ColumnTransformer(transformers=t_features, remainder='passthrough')
    else:
        raise ValueError(
            f"preprocesor debe ser -1 o 1; recibido: {preprocesor}. "
            "Valores soportados: -1=passthrough, 1=dummy+TE."
        )
    return preprocessor

class LGBMModel():
    
        def __init__(self, cols_for_model, hyperparams, search_hip=False, sampling_th=0.5, preprocesor_num=-1, sampling_method='under'):
            """
            Initializes the LGBMModel.

            Args:
                cols_for_model (list): The columns to be used for modeling.
                hyperparams: The hyperparameters for the LGBMClassifier.
                search_hip (bool): Flag indicating whether to perform hyperparameter search.
                sampling_th (float): The sampling threshold.
                preprocesor_num (int): The preprocessor number.
                sampling_method (str): The sampling method ('over' or 'under').
            """
            self.cols_for_model = cols_for_model
            self.sampling_th = sampling_th
            self.preprocesor_num = preprocesor_num
            self.sampling_method = sampling_method
            self.search_hip = search_hip
            self.hyperparams = hyperparams
        
        def build_pipeline_preproceso_model(self):
            preprocessor = get_preprocesor(self.preprocesor_num)
            lgbm_model_search = LGBMClassifier(random_state=314, metric='None',  n_estimators=1000)
            if  self.sampling_method  == 'over':
                over = RandomOverSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(preprocessor,over,lgbm_model_search)
            elif self.sampling_method  == 'under':
                under = RandomUnderSampler(sampling_strategy=self.sampling_th,random_state=40)
                return make_pipeline(preprocessor,under,lgbm_model_search)
            else:
                return make_pipeline(preprocessor,lgbm_model_search)
        
        def train(self,df_train,y_train,df_val=None,y_val=None):
            if  df_val is None:
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.2, random_state=42)
            
            pipe_preproceso_model = self.build_pipeline_preproceso_model()
            
            preprocessor_features = pipe_preproceso_model.steps[0][1]
            # print(preprocessor_features)
            preprocessor_features.fit(df_train[self.cols_for_model], y_train)
            df_val_tra = preprocessor_features.transform(df_val[self.cols_for_model])
            # print(df_val_tra.shape)
            # print(df_val_tra.columns)
            if self.search_hip:
                self.best_score_, self.hyperparams = self.find_hyp_lgbm_model(df_train[self.cols_for_model],y_train,df_val_tra,y_val,pipe_preproceso_model)
                
            params = self.hyperparams
            fit_params = {
                    'eval_metric' : ['auc'],
                    "eval_set": [(df_val_tra, y_val)],
                    "eval_names": ["valid"],
                    "callbacks": [early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False)],
                    "categorical_feature": "auto",
                    "feature_name": "auto",
                }
            new_fit_params = {'lgbmclassifier__' + key: fit_params[key] for key in fit_params}
            
            if params is not None:
                pipe_preproceso_model.set_params(**params)
            pipe_preproceso_model.fit(df_train[self.cols_for_model], y_train, **new_fit_params)
        
            return pipe_preproceso_model
            
        def find_hyp_lgbm_model(self, X_train,y_train,X_val,y_val,imba_pipeline):
                fit_params = {
                     'eval_metric' : ['auc'],
                    "eval_set": [(X_val, y_val)],
                    "eval_names": ["valid"],
                    "callbacks": [early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False)],
                    "categorical_feature": "auto",
                    "feature_name": "auto",
                }

                param_test ={
                    'num_leaves': sp_randint(6, 50), 
                     'max_bin': sp_randint(60, 255),
                     'max_depth': sp_randint(5, 20),
                     'min_child_samples': sp_randint(100, 500), 
                     'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                     'subsample': sp_uniform(loc=0.2, scale=0.8), 
                     'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                     'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                     'reg_lambda': [0, 1e-5, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 50, 100],
                     'scale_pos_weight':[1,5,20,100],
                #             'is_unbalance':[True,False],
                     'learning_rate':sp_uniform(loc=0.01, scale=0.1),
                     'subsample_freq': sp_randint(5, 20)
                    }

                new_params = {'lgbmclassifier__' + key: param_test[key] for key in param_test}
                new_fit_params = {'lgbmclassifier__' + key: fit_params[key] for key in fit_params}

                random_imba = RandomizedSearchCV(estimator = imba_pipeline, 
                                       param_distributions = new_params, 
                                       cv = 3, 
            #                            scoring = 'average_precision',
                                       scoring= 'roc_auc',
                                       n_jobs = 35,
                                       n_iter = 60,
                                       refit=True,
                                       random_state = 314)
                random_imba.fit(X_train, y_train, **new_fit_params);
                return random_imba.best_score_, random_imba.best_params_
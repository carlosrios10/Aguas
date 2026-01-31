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
    
    if preprocesor==-1:
        preprocessor = ColumnTransformer(transformers= [],remainder='passthrough')
        
    if preprocesor==2:
        # Actividad 
        pipe_actividad = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('a_dummy',ToDummy(['distrito']))
                ])

        # Segmento Tarifa
        pipe_tarifa = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('tarifa_te',TeEncoder(['tipo_tarifa'],w=20))
                ])
        
        pipe_distrito = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.001)),
                    ('distrito_te',TeEncoder(['id_distrito'],w=20))
                ])
        
        vars_enc = ['id_cant_fases','id_tarifa']
        t_features = [
            ('var_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), vars_enc),
            # ('a_dummy',ToDummy(['clasif_tarifa'])),
            # ('material_isntalacion_te', TeEncoder(['material_instalacion'],w=10), ['material_instalacion']),
            # ('actividad_cr_dummy', pipe_actividad, ['distrito']),
            ('ditrito_cr_te', pipe_distrito, ['id_distrito']),
            ('area_peligrosa_binary', FunctionTransformer(), 'area_peligrosa'),
            ]

        preprocessor = ColumnTransformer(transformers= t_features,remainder='passthrough')
        
    if preprocesor==3:
        pipe_tarifa = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.003)),
                    ('te',TeEncoder(['id_tarifa'],w=1))
                ])
        pipe_marca_medidor = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.03)),
                    ('te',TeEncoder(['marca_medidor'],w=1))
                ])
        pipe_distrito = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['id_distrito'],w=20))
                ])
        pipe_tipo_medidor = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['id_tipo_medidor'],w=20))
                ])
        pipe_id_ciu = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.002)),
                    ('te',TeEncoder(['id_ciu'],w=20))
                ])
        vars_dummy = ['tip_instalacion','tipo_cliente']
        t_features = [
            ('dummy_var', ToDummy(vars_dummy), vars_dummy),
            ('pipe_tarifa', pipe_tarifa, ['id_tarifa']),
            ('pipe_marca_medidor', pipe_marca_medidor, ['marca_medidor']),
            ('pipe_distrito', pipe_distrito, ['id_distrito']),
            ('pipe_tipo_medidor', pipe_tipo_medidor, ['id_tipo_medidor']),
            ('pipe_id_ciu', pipe_id_ciu, ['id_ciu']),
            ('area_peligrosa_binary', FunctionTransformer(binary_area_peligrosa), 'area_peligrosa'),
            ]

    if preprocesor == 4:
        # Actividad 
        pipe_tipo_servicio = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['tipo_de_servicio'],w=50))
                ])
        
        pipe_modelo_medidor = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                    ('te',TeEncoder(['modelo_medidor'],w=50))
                ])
        
        vars_enc = ["localidad", "colonia"]
        vars_dummy = ['estado_medidor','tiene_vida_util_1']
        t_features = [
            ('enc_var', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), vars_enc),
            ('dummy_var', ToDummy(vars_dummy), vars_dummy),
            ('p_tipo_servicio', pipe_tipo_servicio, ['tipo_de_servicio']),
            ('p_modelo_medidor', pipe_modelo_medidor, ['modelo_medidor']),
            ]

        preprocessor = ColumnTransformer(transformers= t_features,remainder='passthrough')

    if preprocesor == 5:
        pipe_tipo_serv = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                     ('te',TeEncoder(['tipo_serv_gral'],w=50))
                ])
        
        pipe_marca = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                     ('te',TeEncoder(['marca_gral'],w=50))
                ])
        
        pipe_localidad = Pipeline([
                    ('cardinality_reducer', CardinalityReducer(threshold=0.01)),
                     ('te',ToDummy(['localidad_grl']))
                ])
        vars_enc = ["modelo_gral"]
        t_features = [
            ('enc_var', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), vars_enc),
            ('p_tipo_serv', pipe_tipo_serv, ['tipo_serv_gral']),
             ('p_marca', pipe_marca, ['marca_gral']),
            ('p_localidad', pipe_localidad, ['localidad_grl']),
            ]

        preprocessor = ColumnTransformer(transformers= t_features,remainder='passthrough')

    
    return preprocessor

class LGBMModel():
    
        def __init__(self,cols_for_model,hyperparams,search_hip=False,sampling_th = 0.5,preprocesor_num = 3,sampling_method = 'under'):
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
                 df_train, df_val, y_train, y_val = train_test_split(df_train,y_train, test_size=0.1, random_state=42)
            
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
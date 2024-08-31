import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor

from flood_predictor_model.config.core import config

flood_pred_pipe = Pipeline([

    # Scale features
    ('scaler', StandardScaler()),    
    # selector features
    ('selector', VarianceThreshold()),
    
    # Regressor
    ('model', CatBoostRegressor(depth = config.model_config.depth, 
                       learning_rate = config.model_config.learning_rate,
                       iterations = config.model_config.iterations))
    ])

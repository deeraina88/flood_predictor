"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from flood_predictor_model.predict import make_prediction


def test_make_prediction(sample_input_data):

    result = make_prediction(input_data = sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    
    _predictions = list(predictions)
    y_true = sample_input_data[1]

    mse = mean_squared_error(y_true, _predictions)
    print(mse)
    assert mse < 1

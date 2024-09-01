import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np
import json

from flood_predictor_model import __version__ as _version
from flood_predictor_model.config.core import config
from flood_predictor_model.processing.data_manager import load_pipeline
from flood_predictor_model.processing.data_manager import pre_pipeline_preparation
from flood_predictor_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
flood_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = flood_pipe.predict(validated_data)
        results = {"predictions": np.round(predictions, decimals=3), "version": _version, "errors": errors}
    print(results)
    return results



if __name__ == "__main__":
#1863262,9,7,3,5,6,7,6,7,1,4,2,5,7,5,6,3,9,4,4,5
    data_in = {'id': [1863262],'MonsoonIntensity': [2],'TopographyDrainage': [7],'RiverManagement': [3],'Deforestation': [5],'Urbanization': [6],'ClimateChange': [5],'DamsQuality': [1],'Siltation': [9],'AgriculturalPractices': [3],'Encroachments': [2],'IneffectiveDisasterPreparedness': [13],'DrainageSystems': [2],'CoastalVulnerability': [7],'Landslides': [8],'Watersheds': [8],'DeterioratingInfrastructure': [3],'PopulationScore': [6],'WetlandLoss': [6],'InadequatePlanning': [5],'PoliticalFactors': [5]}

    make_prediction(input_data = data_in)
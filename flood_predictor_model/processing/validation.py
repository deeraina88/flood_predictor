import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from flood_predictor_model.config.core import config
from flood_predictor_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    MonsoonIntensity: int
    TopographyDrainage: int
    RiverManagement: int
    Deforestation: int
    Urbanization: int
    ClimateChange: int
    DamsQuality: int
    Siltation: int
    AgriculturalPractices: int
    Encroachments: int
    IneffectiveDisasterPreparedness: int
    DrainageSystems: int
    CoastalVulnerability: int
    Landslides: int
    Watersheds: int
    DeterioratingInfrastructure: int
    PopulationScore: int
    WetlandLoss: int
    InadequatePlanning: int
    PoliticalFactors: int


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
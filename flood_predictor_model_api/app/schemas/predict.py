from typing import Any, List, Optional
import datetime
import numpy as np
from pydantic import BaseModel
from flood_predictor_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "id": 1863262,
                        "MonsoonIntensity": 9,
                        "TopographyDrainage": 7,
                        "RiverManagement": 3,
                        "Deforestation": 5,
                        "Urbanization": 6,
                        "ClimateChange": 5,
                        "DamsQuality": 1,
                        "Siltation": 9,
                        "AgriculturalPractices": 3,
                        "Encroachments": 2,
                        "IneffectiveDisasterPreparedness": 13,
                        "DrainageSystems": 2,
                        "CoastalVulnerability": 7,
                        "Landslides": 8,
                        "Watersheds": 8,
                        "DeterioratingInfrastructure": 3,
                        "PopulationScore": 6,
                        "WetlandLoss": 6,
                        "InadequatePlanning": 5,
                        "PoliticalFactors": 5
                    }

                ]
            }
        }

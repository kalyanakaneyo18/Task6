"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel, Field, field_validator


class PropertyRequest(BaseModel):
    """Raw property information accepted by POST /predict."""

    area: float = Field(gt=0, description="Property area (square feet)")
    bedrooms: int = Field(ge=1, le=5, description="Number of bedrooms")
    bathrooms: int = Field(ge=1, le=3, description="Number of bathrooms")
    age: int = Field(ge=0, le=49, description="Property age in years")
    location: str = Field(description="Location: Rural, Suburb, or City Center")
    property_type: str = Field(description="Property type: House, Villa, or Apartment")

    @field_validator("location", "property_type")
    @classmethod
    def strip_and_title(cls, v: str) -> str:
        return v.strip().title()


class PredictionResponse(BaseModel):
    predicted_price: float

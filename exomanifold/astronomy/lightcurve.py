"""
Light curve representation for ExoManifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__= [
    "ExoLightCurve",
]

@dataclass(frozen=True, slots = True)
class ExoLightCurve:
    """
    Immutable representation of a single astronomical light curve.
    """

    time: NDArray[np.float64]

    flux: NDArray[np.float64]

    flux_err: NDArray[np.float64] | None = None

    quality: NDArray[np.int32] | None = None

    target: str = ""

    mission: str = ""

    cadence: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self)->None:

        if self.time.ndim != 1:
            raise ValueError("time must be one-dimensional.")
        
        if self.flux.ndim != 1:
            raise ValueError("flux must be one-dimensional.")
        
        if len(self.time) != len(self.flux):
            raise ValueError("time and flux must have the same length.")
        

        if self.flux_err is not None:

            if len(self.flux_err) != len(self.time):

                raise ValueError("flux_err has incorrect length.")
        
        if self.quality is not None:

            if len(self.quality) != len(self.time):
                raise ValueError("quality has incorrect length.")
            
    
    def __len__(self)->int:
        return len(self.time)
    
    def __repr__(self)-> str:
        return (
            "ExoLightCurve("
            f"target = '{self.target}',"
            f"mission = '{self.mission}'"
            f"samples = '{len(self)}'"
        )
    
    def copy(self) -> "ExoLightCurve":
        return ExoLightCurve(
            time = self.time.copy(),
            flux = self.flux.copy(),
            flux_err = None if self.flux_err is None else self.flux_err.copy(),
            quality= None if self.quality is None else self.quality.copy(),
            target=self.target,
            mission = self.mission,
            cadence=self.cadence,
            metadata=self.metadata.copy()
        )
    
    def to_numpy(self)-> NDArray[np.float64]:
        return self.flux.copy()
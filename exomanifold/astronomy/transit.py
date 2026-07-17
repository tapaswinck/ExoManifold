"""
Transit detection utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from astropy.timeseries import BoxLeastSquares

from exomanifold.astronomy.lightcurve import ExoLightCurve


__all__ = [
    "TransitEvent",
    "TransitSearchResult",
    "TransitDetector"
]


#------------
#TransitEvent
#------------

@dataclass(frozen=True, slots=True)
class TransitEvent:
    """
    Represents a detected transit signal.
    """

    period: float
    epoch: float
    duration: float
    depth: float
    snr: float
    power: float

    def __post_init__(self)-> None:

        if self.period <= 0:
            raise ValueError("period must be positive.")
        
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        
        if self.depth < 0:
            raise ValueError("depth cannot be negative.")
        
        if self.snr < 0:
            raise ValueError("snr cannot be negative.")
        
        if self.power < 0:
            raise ValueError("power cannot be negative.")
        
#-------------------
#TransitSearchResult
#-------------------

@dataclass(frozen=True,slots = True)
class TransitSearchResult:
    """
    Stores the result of a transit search.
    """

    best_event: TransitEvent
    events: list[TransitEvent]

    period_grid: NDArray[np.float64]
    power: NDArray[np.float64]

    algorithm: str
    
    metadata: dict = field(default_factory=dict)

    def __post_init__(self)-> None:
        
        period_grid = np.asarray(self.period_grid, dtype = float)
        power = np.asarray(self.power, dtype=float)

        if period_grid.ndim != 1:
            raise ValueError("period_grid must be one-dimensional.")
        
        if power.ndim != 1:
            raise ValueError("power must be one dimensional.")
        
        if len(period_grid) != len(power):
            raise ValueError("period_grid and power must have same length.")
        
        object.__setattr__(self, "period_grid", period_grid)
        object.__setattr__(self, "power", power)

    def __len__(self)-> int:
        return len(self.period_grid)
    


#---------------
#TransitDetector
#---------------

class TransitDetector:
    """
    Search for transiting exoplanets.
    """

    def __init__(
            self,
            minimum_period: float = 0.5,
            maximum_period: float = 30.0,
            n_periods: int = 1000,
            oversample: int =5,
            duration_grid: NDArray[np.float64] | None = None
    )-> None:
        
        if minimum_period <= 0:
            raise ValueError("minimum_period must be positive.")
        
        if maximum_period <= minimum_period:
            raise ValueError("maximum_period must not be lesser than minimum_period.")
        
        if n_periods < 2:
            raise ValueError("n_periods must be atleast 2.")
        
        if oversample < 1:
            raise ValueError("oversample must be atleast one.")
        
        self.minimum_period = float(minimum_period)
        self.maximum_period = float(maximum_period)
        self.n_periods = int(n_periods)
        self.oversample = int(oversample)

        if duration_grid is None:
            duration_grid = np.array([0.05, 0.10, 0.20, 0.30], dtype = float)

        duration_grid = np.asarray(duration_grid, dtype = float)

        if duration_grid.ndim != 1:
            raise ValueError("duration_grid must be one-dimensional.")
        
        if np.any(duration_grid <= 0):
            raise ValueError("duration grid must contain positive values.")


        self.duration_grid = duration_grid


    def compute_period_grid(
            self
    )->NDArray[np.float64]:
        """
        Compute the trial periods.
        """

        return np.linspace(
            self.minimum_period,
            self.maximum_period,
            self.n_periods,
            dtype = float
        )
    
    def search(
            self,
            curve: ExoLightCurve
    )-> TransitSearchResult:
        """
        Search for periodic transit signals using Box Least Squares.
        """
        self._validate_curve(curve)

        model = BoxLeastSquares(
            curve.time,
            curve.flux,
            dy = curve.flux_err
        )

        periods = self.compute_period_grid()

        result = model.power(
            periods,
            self.duration_grid,
            oversample = self.oversample
        )

        best = np.argmax(result.power)

        event = TransitEvent(
            period = float(result.period[best]),
            epoch = float(result.transit_time[best]),
            duration = float(result.duration[best]),
            depth = float(result.depth[best]),
            snr = float(result.depth_snr[best]),
            power = float(result.power[best])
        )

        return TransitSearchResult(
            best_event=event,
            events = [event],
            period_grid=result.period,
            power = result.power,
            algorithm="BLS"
        )
    
    def fit(
            self,
            curve: ExoLightCurve
    )-> "TransitDetector":
        """
        Compatibility with scikit-learn style API.
        """
        _ = curve

        return self
    
    def fit_search(
            self,
            curve: ExoLightCurve
    )-> TransitSearchResult:
        """
        Fit and search.
        """

        self.fit(curve)

        return self.search(curve)
    

    def __repr__(self)-> str:
        return (
            f"TransitDetector("
            f"minimum_period = {self.minimum_period},"
            f"maximum_period = {self.maximum_period},"
            f"n_periods = {self.n_periods}"
        )

    def _validate_curve(
            self,
            curve: ExoLightCurve
    )->None:
        """
        Validate an input light curve.
        """

        if len(curve) < 10:
            raise ValueError("Light curve must contain at least 10 samples.")
        
        if np.any(~np.isfinite(curve.time)):
            raise ValueError("time contains no-finite values.")
        
        if np.any(~np.isfinite(curve.flux)):
            raise ValueError("flux contains non-finite values.")

        if curve.flux_err is not None:
            if np.any(~np.isfinite(curve.flux_err)):
                raise ValueError("flux_err contains non-finite values.")


        if np.any(np.diff(curve.time) <= 0):
            raise ValueError("time value must strictly be increasing.")





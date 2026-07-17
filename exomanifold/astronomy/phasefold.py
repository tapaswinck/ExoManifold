"""
Phase-folded light curve class.
"""

from __future__ import annotations


from dataclasses import dataclass, field, replace
from .lightcurve import ExoLightCurve

import numpy as np
from numpy.typing import NDArray

from exomanifold.utils.validation import (
    check_vector,
)


__all__ = [
    "PhaseFoldedLightCurve",
    "PhaseFolder"
]

@dataclass(frozen=True, slots = True)
class PhaseFoldedLightCurve:
    """
    Immutable phase-folded light curve.

    Parameters
    ----------
    phase: ndarray
        Orbital phase
    
    flux: ndarray
        Flux values

    flux_err: ndarray | None, optional
        Flux uncertainties
    
    target: str | None
        target identifier
    
    mission: str | None
        Kepler, TESS, etc
    
    period: float
        Orbital period

    epoch: float
        Transit epoch

    metadata: dict
        additional metadata
    """

    phase: NDArray[np.float64]
    flux: NDArray[np.float64]

    flux_err: NDArray[np.float64] | None = None
    
    target: str | None = None
    mission: str | None = None

    period: float = 0.0
    epoch: float = 0.0

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = np.asarray(self.phase, dtype=float)
        flux = np.asarray(self.flux, dtype=float)

        phase = check_vector(phase, name="phase")
        flux = check_vector(flux, name="flux")

        if len(phase) != len(flux):
            raise ValueError("phase and flux must have the same length.")

        if self.flux_err is not None:
            flux_err = np.asarray(self.flux_err, dtype=float)
            flux_err = check_vector(flux_err, name="flux_err")

            if len(flux_err) != len(phase):
                raise ValueError(
                    "flux_err must have the same length as phase."
                )

            object.__setattr__(self, "flux_err", flux_err)

        if self.period <= 0:
            raise ValueError("period must be positive.")

        if np.any((phase < 0.0) | (phase > 1.0)):
            raise ValueError("phase values must lie between 0 and 1.")

        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "flux", flux)

    
    @property
    def n_samples(self)-> int:
        """
        Number of observations.  
        """
        return len(self.phase)
    
    @property
    def has_flux_errors(self)-> bool:
        """
        True if flux uncertainties exist.
        """

        return self.flux_err is not None
    
    def __len__(self) -> int:
        return len(self.phase)
    
    def sort(self) -> "PhaseFoldedLightCurve":
        """
        return a copy sorted by phase.
        """

        order = np.argsort(self.phase)

        return replace(
            self,
            phase = self.phase[order],
            flux = self.flux[order],
            flux_err = (
                None
                if self.flux_err is None
                else
                self.flux_err[order]
            )
        )
    
    def copy(self) -> "PhaseFoldedLightCurve":
        """
        Return a deep copy.
        """
        return replace(
            self,
            phase = self.phase.copy(),
            flux = self.flux.copy(),
            flux_err = (
                None
                if self.flux_err is None
                else
                self.flux_err.copy()
            ),
            metadata = self.metadata.copy()
        )
class PhaseFolder:
    """
    Phase-fold astronomical light curves.

    Parameters
    ----------
    period: float
        Orbital period
    
    epoch: float
        Reference transit epoch
        
    bins: int, default = 200
        Number of phase bins
    
    wrap_phase: bool, default = True
        Wrap phase into the interval [0,1)
    """

    def __init__(
            self,
            period: float,
            epoch: float,
            bins: int = 200,
            wrap_phase: bool = True
    )-> None:
        
        if period <= 0:
            raise ValueError("period must be positive.")
        
        if bins <= 1:
            raise ValueError("bins must greater than one.")
        
        self.period = float(period)
        self.epoch = float(epoch)
        self.bins = int(bins)
        self.wrap_phase = wrap_phase

    def compute_phase(
            self,
            curve: ExoLightCurve,
    )-> NDArray[np.float64]:
        """
        Compute orbital phase for every observation.

        Parameters
        ----------
        curve: ExoLightCurve
            Input light curve
        
        Returns
        -------
        ndarray
            Phase values in the interval [0,1)
        """

        phase = (curve.time  - self.epoch) / self.period

        if self.wrap_phase:
            phase = np.mod(phase, 1.0)

        return phase.astype(np.float64)
    
    def sort_phase(
            self,
            phase: NDArray[np.float64],
            flux: NDArray[np.float64],
            flux_err: NDArray[np.float64] | None = None
    )-> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None
    ]:
        """
        Sort phase and associated arrays by increasing phase.

        Parameters
        ----------
        phase: ndarray
            Orbital phases.

        flux: ndarray
            Flux values
        
        flux_err: ndarray | None, optional
            Flux uncertainties

        Returns
        -------
        tuple
            sorted phase, flux, and flux uncertainties
        """
        phase = check_vector(phase, name = "phase")
        flux = check_vector(flux,name="flux")

        if len(phase) != len(flux):
            raise ValueError("phase and flux must have the same length.")
        
        if flux_err is not None:
            flux_err = check_vector(flux_err, name="flux_err")

            if len(flux_err) != len(phase):
                raise ValueError("flux_err must have the same length as phase.")

        order = np.argsort(phase)

        sorted_phase = phase[order]
        sorted_flux = flux[order]

        sorted_flux_err = (
            None
            if flux_err is None
            else
            flux_err[order]
        )

        return (
            sorted_phase,
            sorted_flux,
            sorted_flux_err
        )    
    
    def bin_phase(
            self,
            phase: NDArray[np.float64],
            flux: NDArray[np.float64],
            flux_err: NDArray[np.float64] | None = None
    )-> PhaseFoldedLightCurve:
        """
        Bin a phase-folded light curve.

        Parameters
        ----------
        phase: ndarray
            Orbital phase values

        flux: ndarray
            Flux values

        flux_err: ndarray | None, optional
            Flux uncertainties

        Returns
        -------
        PhaseFoldedLightCurve
            Binnedd phase-folded light curve.
        """

        phase = check_vector(phase, name="phase")
        flux = check_vector(flux, name="flux")

        if len(phase) != len(flux):
            raise ValueError("phase and flux must have the same length.")
        
        if flux_err is not None:
            flux_err = check_vector(flux_err, name = "flux_err")

            if len(flux_err) != len(phase):
                raise ValueError("flux_err must have the same length as phase.")
            

        #-----------------------------------
        #Construct equallt spaced phase bins
        #-----------------------------------

        edges = np.linspace(0.0, 1.0, self.bins + 1)

        indices = np.digitize(phase, edges, right=False) - 1

        #phase == 1.0 belongs in the last bin
        indices = np.clip(indices, 0, self.bins -1)

        phase_bins: list[float] = []
        flux_bins: list[float] = []

        flux_err_bins: list[float] | None

        if flux_err is None:
            flux_err_bins = None
        else:
            flux_err_bins = []

        #-------------------------------
        #Average values inside each bins
        #-------------------------------

        for i in range(self.bins):

            mask = indices == i

            if not np.any(mask):
                continue

            phase_bins.append(float(np.mean(phase[mask])))
            flux_bins.append(float(np.mean(flux[mask])))

            if flux_err is not None:
                assert flux_err is not None

                flux_err_bins.append(float(np.mean(flux_err[mask])))
                

        return PhaseFoldedLightCurve(
            phase = np.asarray(phase_bins),
            flux = np.asarray(flux_bins),
            flux_err = (
                None
                if flux_err_bins is None
                else
                np.array(flux_err_bins)
            ),
            period = self.period,
            epoch = self.epoch
        )

from __future__ import annotations

from dataclasses import replace

import numpy as np

from exomanifold.astronomy.lightcurve import ExoLightCurve
__all__ = [
    "ExoProcessor"
]

class ExoProcessor:
    """
    Preprocess astronomical light curves.

    Every preprocessing method returns a new ExoLightCurve object.
    The original object is never modified.
    """

    def __init__(self)-> None:
        "Create a preprocessor"
        pass

    def remove_nans(
            self,
            curve: ExoLightCurve,
    )-> ExoLightCurve:
        """
        Remove observations containing NaN values.

        Parameters
        ----------
        curve: ExoLightCurve
            Input light curve

        Returns
        -------
        ExoLightCurve
            New light curve with NaN values removed.
        """

        mask = np.isfinite(curve.time)
        mask &= np.isfinite(curve.flux)

        if curve.flux_err is not None:
            mask &= np.isfinite(curve.flux_err)

        
        time = curve.time[mask]
        flux = curve.flux[mask]

        flux_err = (
            None
            if curve.flux_err is None
            else curve.flux_err[mask]
            )
        
        quality = (
            None
            if curve.quality is None
            else curve.quality[mask]
        )

        return replace(
            curve,
            time = time,
            flux = flux,
            flux_err = flux_err,
            quality = quality
        )
    
    def remove_quality_flags(
        self,
        curve: ExoLightCurve,
        ) -> ExoLightCurve:
        """
        Remove observations with non-zero quality flags.

        Parameters
        ----------
        curve : ExoLightCurve
            Input light curve.

        Returns
        -------
        ExoLightCurve
            Light curve containing only observations with quality == 0.
        """

        if curve.quality is None:
            return curve

        mask = curve.quality == 0

        return replace(
            curve,
            time=curve.time[mask],
            flux=curve.flux[mask],
            flux_err=(
                None
                if curve.flux_err is None
                else curve.flux_err[mask]
            ),
            quality=curve.quality[mask],
        )
    
    def normalize_flux(
        self,
        curve: ExoLightCurve,
    ) -> ExoLightCurve:
        """
        Normalize the flux by its median value.

        Parameters
        ----------
        curve : ExoLightCurve
            Input light curve.

        Returns
        -------
        ExoLightCurve
            Normalized light curve.
        """

        median = np.median(curve.flux)

        if median == 0:
            raise ValueError(
                "Cannot normalize a light curve with zero median flux."
            )

        return replace(
            curve,
            flux=curve.flux / median,
            flux_err=(
                None
                if curve.flux_err is None
                else curve.flux_err / median
            ),
        )
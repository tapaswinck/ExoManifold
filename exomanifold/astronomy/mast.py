"""
Interface to the MAST archive using lightkurve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightkurve as lk
import numpy as np

from exomanifold.astronomy.lightcurve import ExoLightCurve

__all__ = [
    "MASTClient",
]

class MASTClient:
    """
    Client for querying and downloading light curves from MAST.

    Parameters
    ----------

    cache_dir: Path | None, optional
        Directory used by Lightkurve for caching downloads.
        If None, Lightkurve's default cache is used.

    mission: str, default="Kepler"
        Default mission to query.
    """

    def __init__(
            self,
            cache_dir: Path | None = None,
            mission: str = "Kepler"
    )-> None:
        
        self.cache_dir = cache_dir
        self.mission = mission


    def search(
            self,
            target: str,
            mission: str | None = None,
    ):
        """
        Search the MAST archive.
        """
        mission = mission or self.mission

        return lk.search_lightcurve(
            target,
            mission = mission
        )
    
    def download_lightcurve(
            self,
            target: str,
            mission: str | None = None
    )->ExoLightCurve:
        """
        Download the first available light curve.
        """

        mission = mission or self.mission

        search = self.search(
            target,
            mission
        )

        if len(search) == 0:
            raise ValueError(f"No light curves found for '{target}'.")
        
        lc = search.download()

        return ExoLightCurve(
            time = np.asarray(lc.time.value),
            flux = np.asarray(lc.flux.value),
            flux_err = None if lc.flux_err is None else np.asarray(lc.flux_err.value),
            quality=None if lc.quality is None else np.asarray(lc.quality),
            target = target,
            mission = mission,
            metadata = dict(lc.meta)
        )
    

    def download_multiple(
            self,
            targets: list[str],
    )-> list[ExoLightCurve]:
        
        curves = []

        for target in targets:
            curves.append(
                self.download_lightcurve(target)
            )

        return curves

    def cache_info(self)-> Path | None:
        """
        Return the configured cache directory.
        """        
        return self.cache_dir
    
    def clear_cache(self)-> None:
        """
        Placeholder for cache management.
        """

        raise NotImplementedError("Cache clearing is not implemented yet.")
    
    
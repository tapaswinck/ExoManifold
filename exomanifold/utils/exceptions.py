"""
Custom exceptions used throughout ExoManifold.
"""

class ExoManifoldError(Exception):

    """
    Base Exception for ExoManifold.
    """

class NotFittedError(ExoManifoldError):
    """
    Raised when an estimator is used before fitting.
    """

class InvalidLightCurveError(ExoManifoldError):
    """
    Raised when a light curve cannot be processed.
    """

class DatasetError(ExoManifoldError):
    """
    Raised when dataset construction fails.
    """


    
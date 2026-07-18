"""
Classification metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report"
]


def accuracy_score(
        y_true: NDArray[np.int_],
        y_pred: NDArray[np.int_]
)->float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true: ndarray
        Ground-truth labels.
    
    y_pred: ndarray
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly classified samples.
    """

    y_true = np.asarray(y_true)
    y_pred =np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")
    
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")
    
    return float(np.mean(y_true == y_pred))

def confusion_matrix(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Compute a binary confusion matrix.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth binary labels (0 or 1).

    y_pred : ndarray
        Predicted binary labels (0 or 1).

    Returns
    -------
    ndarray of shape (2, 2)

        [[TN, FP],
         [FN, TP]]

    Raises
    ------
    ValueError
        If the inputs are invalid or contain labels other than 0 and 1.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if y_pred.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    labels = np.unique(np.concatenate((y_true, y_pred)))

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(
            "confusion_matrix currently only supports binary classification."
        )

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=int,
    )

def precision_score(
          y_true: NDArray[np.int_],
          y_pred: NDArray[np.int_]
)-> float:
     """
     Compute binary classification precision.
     """
     cm = confusion_matrix(y_true, y_pred)

     tp = cm[1,1]
     fp = cm[0,1]

     denominator = tp + fp
     if denominator == 0:
          return 0.0
     
     return float(tp / denominator)

def recall_score(
          y_true: NDArray[np.int_],
          y_pred: NDArray[np.int_]
)-> float:
     """
     Compute binary classification recall.
     """

     cm = confusion_matrix(y_true, y_pred)

     tp = cm[1,1]
     fn = cm[1,0]

     denominator = tp + fn

     if denominator ==0:
          return 0.0
     
     return float(tp / denominator)

def f1_score(
          y_true: NDArray[np.int_],
          y_pred: NDArray[np.int_]
)-> float:
     """
     Compute binary classification F1 score.
     """

     precision = precision_score(y_true, y_pred)
     recall = recall_score(y_true, y_pred)

     denominator = precision + recall

     if denominator == 0:
          return 0.0
     return float(
          2.0 * precision * recall / denominator
     )


def classification_report(
          y_true: NDArray[np.int_],
          y_pred: NDArray[np.int_]
)-> dict[str , float | int]:
     """
     Compute a summary of binary classification metrics.
     
     Returns
     -------
     dict
        Dictionary containing accuracy, precision, 
        recall, F1 score and support.
     """
     return {
          "accuracy": accuracy_score(y_true, y_pred),
          "precision": precision_score(y_true, y_pred),
          "recall": recall_score(y_true, y_pred),
          "f1_score": f1_score(y_true, y_pred),
          "support": len(y_true)
     }
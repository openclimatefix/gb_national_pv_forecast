"""Define the metrics functions themselves, and utility functions for running pipelines.

Please see the docstring of `run_all_pipelines` for more information.
"""

from numbers import Number
from typing import Callable, Union

import pandas as pd


def run_all_pipelines(
    metrics_pipelines: dict[str, tuple[Callable, ...]],
    predictions: pd.Series,
    actual: pd.Series,
) -> dict[str, Union[Number, pd.Series]]:
    """Run all the metric pipelines given as keys in `metrics_pipelines`.

    Each `metrics_pipeline` is a tuple of one or more `callable` objects.

    For example:

    ```python
    # Define re-usable metrics pipelines:
    metrics_pipelines: dict[str, tuple[Callable, ...]] = {
        "MAE in MW, ignoring night": (
            # Multiply the `predictions` and the `actual` by national installed PV capacity:
            Denormalize(national_pv_capacity_mwp_per_target_datetime_utc),

            # Drop any values that occur at "night". Where "night" is defined as when the Sun
            # is below a particular angle in the sky:
            IgnoreNight(latitude, longitude, threshold_for_sun_angle_in_degrees=-5),

            # Compute the absolute error for each timestep:
            predictions_and_actual_kwargs_to_tuple,
            np.subtract,
            np.abs,

            # Compute the mean across all timesteps:
            np.mean,
        ),
        "NMAE per month, including night": (
            # Compute the absolute error for each timestep:
            absolute_error,

            # Group by month:
            partial(
                pd.Series.groupby,
                by=pd.Grouper(freq="M"),
            ),

            # Compute the mean absolute error per month:
            pd.core.groupby.GroupBy.mean,
        ),
        "NMAE per hour, ignoring night": (
            IgnoreNight(latitude, longitude, threshold_for_sun_angle_in_degrees=-5),
            absolute_error,

            # Group by hour of day:
            GroupBy(pd.Grouper(freq="H")),

            # Compute the mean absolute error per hour of the day:
            mean,
        )
    }

    # Run the pipelines:
    metrics_results: dict[str, Union[Number, pd.Series, pd.DataFrame]] = run_all_pipelines(
        metrics_pipelines, predictions, actual)
    ```

    Args:
        metrics_pipelines: A dict where each key is a human-readable name for the metric pipeline,
            and each value is a metric pipeline defined by a tuple of callables.
        predictions: a `pandas.DataFrame`:
            Index: `target_datetime_utc`
            Columns:
            - `step`  # The timedelta between the time the forecast was created (`t0`)
                      # and the `target_datetime_utc`.
            - `forecast_pv_yield`  # Values must be in the range [0, 1].
        actual: a `pd.Series`:
            Index: `target_datetime_utc`
            Values: The `actual_pv_yield` (in the range [0, 1])

    Returns: A dict with the same keys as `metrics_pipelines`. The values are the result of each
        metrics pipeline.
    """
    metrics_results: dict[str, Union[Number, pd.Series, pd.DataFrame]] = {}
    for metric_name, pipeline in metrics_pipelines.items():
        metrics_results[metric_name] = run_pipeline(pipeline, predictions, actual)
    return metrics_results


def run_pipeline(
    pipeline: tuple[Callable, ...], predictions: pd.DataFrame, actual: pd.Series
) -> Union[Number, pd.Series, pd.DataFrame]:
    """Run a single metrics pipeline."""
    # `output` starts as a dict holding the predictions and actual data, and `output` will be
    # transformed by each function in the pipeline.
    output: Union[dict, Number, pd.Series, pd.DataFrame] = dict(
        predictions=predictions, actual=actual
    )
    for function in pipeline:
        if isinstance(output, dict):
            output = function(**output)
        elif isinstance(output, tuple):
            output = function(*output)
        else:
            output = function(output)
    return output


def denormalize(
    predictions: pd.Series, actual: pd.Series, pv_capacity: Union[Number, pd.Series]
) -> dict[str, pd.Series]:
    if isinstance(pv_capacity, pd.Series) and pv_capacity.index.freq != actual.index.freq:
        # Resample pv_capacity to match the frequency of `actual`.
        pv_capacity = pv_capacity.resample(freq=actual.index.freq).ffill()
    predictions_denorm = predictions * pv_capacity
    actual_denorm = actual * pv_capacity

    return dict(predictions=predictions_denorm, actual=actual_denorm)

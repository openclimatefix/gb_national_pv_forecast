"""Fourier encoding, and utils."""
from typing import Final, Union

import numpy as np
import pandas as pd

SECONDS_PER_HOUR: Final[int] = 60 * 60
MINUTES_PER_HOUR: Final[int] = 60
HOURS_PER_DAY: Final[int] = 24
DAYS_PER_YEAR: Final[int] = 365


def fourier_encode_date_time(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode time-of-day and day-of-year as sin and cos features.

    Returns:
        pd.DataFrame with columns: "day_sin", "day_cos",
            "year_sin", "year_cos",
            "year_x4_sin", "year_x4_cos"
    """
    assert isinstance(dt_index, pd.DatetimeIndex)
    assert not dt_index.empty

    # Compute the hour of day as radians, taking into consideration the minutes and seconds.
    # For example, 12:30pm is 12.5 hours through the day.
    hour_of_day = (
        dt_index.hour + (dt_index.minute / MINUTES_PER_HOUR) + (dt_index.second / SECONDS_PER_HOUR)
    )
    hour_of_day_fraction = hour_of_day / HOURS_PER_DAY
    hour_of_day_radians = fraction_to_radians(hour_of_day_fraction)

    # Compute the day of year as radians:
    day_of_year = dt_index.day_of_year + hour_of_day_fraction
    day_of_year_radians = fraction_to_radians(day_of_year / DAYS_PER_YEAR)

    # Return the final DataFrame:
    return pd.DataFrame(
        index=dt_index,
        data={
            "day_sin": np.sin(hour_of_day_radians),
            "day_cos": np.cos(hour_of_day_radians),
            "year_sin": np.sin(day_of_year_radians),
            "year_cos": np.cos(day_of_year_radians),
            "year_x4_sin": np.sin(day_of_year_radians * 4),
            "year_x4_cos": np.cos(day_of_year_radians * 4),
        },
    )


def fraction_to_radians(fraction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert a number in the range [0, 1] to radians."""
    return fraction * 2 * np.pi

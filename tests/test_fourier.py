import io

import numpy as np
import pandas as pd

from pv_forecast import fourier


def test_fraction_to_radians():
    np.testing.assert_almost_equal(fourier.fraction_to_radians(0), 0)
    np.testing.assert_almost_equal(fourier.fraction_to_radians(1), 2 * np.pi)


def test_fourier_encode_date_time():
    date_range = pd.date_range("2020-01-01T00:00", "2020-01-01T00:10", freq="1T")
    encoded = fourier.fourier_encode_date_time(date_range)
    # The CSV string is created using: print(encoded.to_csv(float_format="%.7f"))
    true = pd.read_csv(
        io.StringIO(
            """,day_sin,day_cos,year_sin,year_cos,year_x4_sin,year_x4_cos
2020-01-01 00:00:00,0.0000000,1.0000000,0.0172134,0.9998518,0.0688024,0.9976303
2020-01-01 00:01:00,0.0043633,0.9999905,0.0172253,0.9998516,0.0688501,0.9976270
2020-01-01 00:02:00,0.0087265,0.9999619,0.0172373,0.9998514,0.0688978,0.9976237
2020-01-01 00:03:00,0.0130896,0.9999143,0.0172492,0.9998512,0.0689455,0.9976204
2020-01-01 00:04:00,0.0174524,0.9998477,0.0172612,0.9998510,0.0689932,0.9976171
2020-01-01 00:05:00,0.0218149,0.9997620,0.0172731,0.9998508,0.0690409,0.9976138
2020-01-01 00:06:00,0.0261769,0.9996573,0.0172851,0.9998506,0.0690886,0.9976105
2020-01-01 00:07:00,0.0305385,0.9995336,0.0172970,0.9998504,0.0691364,0.9976072
2020-01-01 00:08:00,0.0348995,0.9993908,0.0173090,0.9998502,0.0691841,0.9976039
2020-01-01 00:09:00,0.0392598,0.9992290,0.0173209,0.9998500,0.0692318,0.9976006
2020-01-01 00:10:00,0.0436194,0.9990482,0.0173329,0.9998498,0.0692795,0.9975973"""
        ),
        index_col=0,
        parse_dates=True,
    )
    true.index.freq = true.index.inferred_freq  # From https://stackoverflow.com/a/69852104/732596
    pd.testing.assert_frame_equal(encoded, true)

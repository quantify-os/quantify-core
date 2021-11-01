import pytest
from jsonschema import ValidationError

from quantify_core.analysis.types import AnalysisSettings


def test_analysis_settings_valid():
    _ = AnalysisSettings(
        {
            "mpl_dpi": 450,
            "mpl_fig_formats": ["svg", "png"],
            "mpl_exclude_fig_titles": False,
            "mpl_transparent_background": False,
            "bla": 123,
        }
    )


def test_analysis_settings_invalid():

    with pytest.raises(ValidationError):
        _ = AnalysisSettings(
            {
                "mpl_fig_formats": ["svg"],
                "mpl_exclude_fig_titles": False,
                "mpl_transparent_background": False,
            }
        )

    with pytest.raises(ValidationError):
        _ = AnalysisSettings(
            {
                "mpl_dpi": "450",
                "mpl_fig_formats": ["svg"],
                "mpl_exclude_fig_titles": False,
                "mpl_transparent_background": False,
            }
        )

    with pytest.raises(ValidationError):
        _ = AnalysisSettings(
            {
                "mpl_dpi": "450",
                "mpl_fig_formats": "svg",
                "mpl_exclude_fig_titles": False,
                "mpl_transparent_background": False,
            }
        )

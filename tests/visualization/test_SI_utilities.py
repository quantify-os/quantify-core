import numpy as np
import matplotlib.pyplot as plt
from quantify.visualization.SI_utilities import SI_prefix_and_scale_factor
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel
from quantify.visualization.SI_utilities import SI_val_to_msg_str
from quantify.visualization.SI_utilities import SafeFormatter, format_value_string
from lmfit.parameter import Parameter


def test_non_si():
    unit = "arb.unit."
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
    assert scale_factor == 1
    assert unit == post_unit


def test_si_scale_factors():
    unit = "V"
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
    assert scale_factor == 1
    assert "" + unit == post_unit

    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5000, unit=unit)
    assert scale_factor, 1 == 1000
    assert "k" + unit == post_unit

    scale_factor, post_unit = SI_prefix_and_scale_factor(val=0.05, unit=unit)
    assert scale_factor == 1000
    assert "m" + unit == post_unit


def test_label_scaling():
    """
    This test creates a dummy plot and checks if the tick labels are
    rescaled correctly
    """
    _, ax = plt.subplots()
    x = np.linspace(-6, 6, 101)
    y = np.cos(x)
    ax.plot(x * 1000, y / 1e5)

    set_xlabel(ax, "Distance", "m")
    set_ylabel(ax, "Amplitude", "V")

    xlab = ax.get_xlabel()
    ylab = ax.get_ylabel()
    assert xlab == "Distance [km]"
    assert ylab == "Amplitude [μV]"


def test_si_val_to_msg_str():
    val, unit = SI_val_to_msg_str(1030, "m")
    assert val == str(1.03)
    assert unit == "km"


BASE_STR = "my_test_values_{:.2f}_{:.3f}"
fmt = SafeFormatter()


def test_safe_formatter():

    fmt_string = fmt.format(BASE_STR, 4, 4.32497)
    assert fmt_string == "my_test_values_4.00_4.325"


def test_safe_formatter_missing():
    fmt_string = fmt.format(BASE_STR, 4, None)
    assert fmt_string == "my_test_values_4.00_~~"
    fmt_custom = SafeFormatter(missing="?")
    fmt_string = fmt_custom.format(BASE_STR, 4, None)
    assert fmt_string == "my_test_values_4.00_?"


def test_safe_formatter_bad_format():
    fmt_string = fmt.format(BASE_STR, 4, "myvalue")
    assert fmt_string == "my_test_values_4.00_!!"

    fmt_custom = SafeFormatter(bad_fmt="!")
    fmt_string = fmt_custom.format(BASE_STR, 4, "myvalue")
    assert fmt_string == "my_test_values_4.00_!"


def test_save_formatter_named_args():
    plot_title = fmt.format(
        "{measurement}\n{timestamp}", timestamp="190101_001122", measurement="test"
    )
    assert plot_title == "test\n190101_001122"


def test_format_value_string():
    tau = Parameter("tau", value=5.123456)
    formatted_string = format_value_string("tau", tau)
    assert formatted_string == r"tau: 5.1235$\pm$NaN "

    tau.stderr = 0.03
    formatted_string = format_value_string("tau", tau)
    assert formatted_string == r"tau: 5.1235$\pm$0.0300 "
    tau.stderr = 0.03


def test_format_value_string_unit_aware():
    tau = Parameter("tau", value=5.123456e-6)
    formatted_string = format_value_string("tau", tau, unit="s")
    assert formatted_string == r"tau: 5.1235$\pm$NaN μs"

    tau.stderr = 0.03e-6
    formatted_string = format_value_string("tau", tau, unit="s")
    assert formatted_string == r"tau: 5.1235$\pm$0.0300 μs"
    tau.stderr = 0.03

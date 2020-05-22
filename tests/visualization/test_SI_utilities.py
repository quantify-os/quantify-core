import numpy as np
import matplotlib.pyplot as plt
from quantify.visualization.SI_utilities import SI_prefix_and_scale_factor
from quantify.visualization.SI_utilities import set_xlabel, set_ylabel
from quantify.visualization.SI_utilities import SI_val_to_msg_str


def test_non_SI():
    unit = 'arb.unit.'
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
    assert scale_factor == 1
    assert unit == post_unit


def test_SI_scale_factors():
    unit = 'V'
    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5, unit=unit)
    assert scale_factor == 1
    assert ''+unit == post_unit

    scale_factor, post_unit = SI_prefix_and_scale_factor(val=5000,
                                                         unit=unit)
    assert scale_factor, 1 == 1000
    assert 'k'+unit == post_unit

    scale_factor, post_unit = SI_prefix_and_scale_factor(val=0.05,
                                                         unit=unit)
    assert scale_factor == 1000
    assert 'm'+unit == post_unit


def test_label_scaling():
    """
    This test creates a dummy plot and checks if the tick labels are
    rescaled correctly
    """
    f, ax = plt.subplots()
    x = np.linspace(-6, 6, 101)
    y = np.cos(x)
    ax.plot(x*1000, y/1e5)

    set_xlabel(ax, 'Distance', 'm')
    set_ylabel(ax, 'Amplitude', 'V')

    xlab = ax.get_xlabel()
    ylab = ax.get_ylabel()
    assert xlab == 'Distance (km)'
    assert ylab == 'Amplitude (μV)'


def test_SI_val_to_msg_str():
    val, unit = SI_val_to_msg_str(1030, 'm')
    assert val == str(1.03)
    assert unit == 'km'


# class test_plot_lmfit_res(unittest.TestCase):

#     def test_plot_model_result(self):
#         def line(a, b, x):
#             return a*x+b

#         a = .1
#         b = 5
#         x = np.linspace(0, 20, 31)
#         y = line(a, b, x)

#         line_model = lmfit.Model(line, independent_vars='x')
#         line_model.set_param_hint('a', value=a)
#         line_model.set_param_hint('b', value=b)
#         params = line_model.make_params()
#         fit_res = line_model.fit(y, x=x, params=params)

#         f, ax = plt.subplots()
#         plot_lmfit_res(fit_res, ax=ax, plot_kws={'color': 'C1'},
#                        plot_init=True, plot_init_kws={'ls': '--'})

import numpy as np
from quantify.utilities.experiment_helpers import (create_plotmon_from_historical, load_settings_onto_instrument)
from quantify.data.handling import set_datadir
from quantify.utilities._tests_helpers import get_test_data_dir


test_datadir = get_test_data_dir()
# Always set datadir before instruments
set_datadir(test_datadir)


# Test that we can successfully load the settings of a dummy instrument
def test_load_settings_onto_instrument():
    def get_func(): return 20

    tuid = '20210319-094728-327-69b211'

    instr = Instrument('DummyInstrument')
    
    # A parameter that is both settable and gettable
    instr.add_parameter('settable_param', initial_value=10, parameter_class=ManualParameter)
    # A parameter that is only gettable
    instr.add_parameter('gettable_param', set_cmd=False, get_cmd=get_func)

    load_settings_onto_instrument(instr, tuid)

    assert instr.get('IDN') == {'vendor': None, 'model': 'DummyInstrument', 'serial': None, 'firmware': None}
    assert instr.get('settable_param') == 5
    assert instr.get('gettable_param') == 20



def test_create_plotmon_from_historical():
    
    tuid = "20200504-191556-002-4209ee"
    plotmon = create_plotmon_from_historical(tuid)

    curves_dict = plotmon._get_curves_config()
    x = curves_dict[tuid]["x0y0"]["config"]["x"]
    y = curves_dict[tuid]["x0y0"]["config"]["y"]

    x_exp = np.linspace(0, 5, 50)
    y_exp = np.cos(np.pi * x_exp) * -1
    np.testing.assert_allclose(x[:50], x_exp)
    np.testing.assert_allclose(y[:50], y_exp)

    cfg = plotmon._get_traces_config(which="secondary_QtPlot")[0]["config"]
    assert np.shape(cfg["z"]) == (11, 50)
    assert cfg["xlabel"] == "Time"
    assert cfg["xunit"] == "s"
    assert cfg["ylabel"] == "Amplitude"
    assert cfg["yunit"] == "V"
    assert cfg["zlabel"] == "Signal level"
    assert cfg["zunit"] == "V"

    plotmon.close()

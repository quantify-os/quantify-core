import numpy as np
import pytest
from qcodes.instrument import Instrument, ManualParameter, InstrumentChannel
from qcodes.utils import validators
from quantify_core.data.handling import set_datadir
from quantify_core.utilities.experiment_helpers import (
    create_plotmon_from_historical,
    load_settings_onto_instrument,
    get_all_parents,
)


@pytest.fixture(scope="function", autouse=False)
def mock_instr(request):
    """
    Set up an instrument with a sub module with the following structure
    """

    def get_func():
        return 20

    instr = Instrument("DummyInstrument")

    # A parameter that is both settable and gettable
    instr.add_parameter(
        "settable_param", initial_value=10, parameter_class=ManualParameter
    )
    # A parameter that is only gettable
    instr.add_parameter("gettable_param", set_cmd=False, get_cmd=get_func)
    # A boolean parameter that is True by defualt
    instr.add_parameter(
        "boolean_param", initial_value=True, parameter_class=ManualParameter
    )
    # A parameter which is already set to None
    instr.add_parameter(
        "none_param",
        initial_value=None,
        parameter_class=ManualParameter,
        vals=validators.Numbers(),
    )
    # A parameter which our function will try to set to None, giving a warning
    instr.add_parameter(
        "none_param_warning",
        initial_value=1,
        parameter_class=ManualParameter,
        vals=validators.Numbers(),
    )
    # A parameter which our function will try to set to a numpy array
    instr.add_parameter(
        "numpy_array_param",
        initial_value=np.array([1]),
        parameter_class=ManualParameter,
        vals=validators.Arrays(),
    )

    def cleanup_instruments():
        instr.close()

    request.addfinalizer(cleanup_instruments)

    return instr


# pylint: disable=redefined-outer-name
def test_load_settings_onto_instrument(tmp_test_data_dir, mock_instr):
    """
    Test that we can successfully load the settings of a dummy instrument
    """
    # Always set datadir before instruments
    set_datadir(tmp_test_data_dir)

    instr = mock_instr

    tuid = "20210319-094728-327-69b211"

    # The snapshot also contains an 'obsolete_param', that is not included here.
    # This represents a parameter which is no longer in the qcodes driver.

    with pytest.warns(
        UserWarning,
        match='Parameter "none_param_warning" of "DummyInstrument" could not be '
        'set to "None" due to error',
    ):
        load_settings_onto_instrument(instr, tuid)

    with pytest.warns(
        UserWarning,
        match="Could not set parameter obsolete_param in DummyInstrument. "
        "DummyInstrument does not possess a parameter named obsolete_param.",
    ):
        load_settings_onto_instrument(instr, tuid)

    assert instr.get("IDN") == {
        "vendor": None,
        "model": "DummyInstrument",
        "serial": None,
        "firmware": None,
    }
    assert instr.get("settable_param") == 5
    assert instr.get("gettable_param") == 20
    assert instr.get("none_param") is None
    assert instr.get("none_param_warning") == 1
    assert not instr.get("boolean_param")
    assert isinstance(instr.get("numpy_array_param"), np.ndarray)
    assert instr.get("numpy_array_param").shape[0] == 4
    assert (instr.get("numpy_array_param") == np.array([0, 1, 2, 3])).all()

    instr.close()


# pylint: disable=redefined-outer-name
def test_load_settings_onto_numpy_param(tmp_test_data_dir, mock_instr):
    """
    Test that we can successfully load the settings of a dummy instrument
    """
    # Always set datadir before instruments
    set_datadir(tmp_test_data_dir)

    instr = mock_instr

    tuid = "20210319-094728-327-69b211"

    # The snapshot also contains an 'obsolete_param', that is not included here.
    # This represents a parameter which is no longer in the qcodes driver.

    load_settings_onto_instrument(instr.numpy_array_param, tuid)

    assert isinstance(instr.get("numpy_array_param"), np.ndarray)
    assert instr.get("numpy_array_param").shape[0] == 4
    assert (instr.get("numpy_array_param") == np.array([0, 1, 2, 3])).all()

    instr.close()


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function", autouse=False)
def mock_instr_nested(request):
    """
    Set up an instrument with a sub module with the following structure

    instr
    -> a
    -> mod_a
        -> b
    -> mod_b
        -> mod_c
            -> c
    """

    instr = Instrument("DummyInstrument")

    instr.add_parameter("a", parameter_class=ManualParameter)

    mod_a = InstrumentChannel(instr, "mod_a")
    mod_a.add_parameter("b", parameter_class=ManualParameter)
    instr.add_submodule("mod_a", mod_a)

    mod_b = InstrumentChannel(instr, "mod_b")
    mod_c = InstrumentChannel(mod_b, "mod_c")
    mod_b.add_submodule("mod_c", mod_c)
    mod_c.add_parameter("c", parameter_class=ManualParameter)

    instr.add_submodule("mod_b", mod_b)

    def cleanup_instruments():
        instr.close()

    request.addfinalizer(cleanup_instruments)

    return instr


def test_get_all_parents(mock_instr_nested):
    """
    Test that we can get all the parent objects of a qcodes instrument, submodule
    or parameter
    """
    parents = get_all_parents(mock_instr_nested.mod_b.mod_c)
    assert parents == [
        mock_instr_nested,
        mock_instr_nested.mod_b,
        mock_instr_nested.mod_b.mod_c,
    ]

    parents = get_all_parents(mock_instr_nested.mod_b.mod_c.c)
    assert parents == [
        mock_instr_nested,
        mock_instr_nested.mod_b,
        mock_instr_nested.mod_b.mod_c,
        mock_instr_nested.mod_b.mod_c.c,
    ]

    parents = get_all_parents(mock_instr_nested)
    assert parents == [mock_instr_nested]


def test_load_settings_onto_instrument_submodules(tmp_test_data_dir, mock_instr_nested):
    """
    Test that we can successfully load the settings of a dummy instrument
    """
    # Always set datadir before instruments
    set_datadir(tmp_test_data_dir)

    # set some random values
    mock_instr_nested.a(23)
    mock_instr_nested.mod_a.b(42)
    mock_instr_nested.mod_b.mod_c.c(23.1)

    # load settings from a dataset
    tuid = "20220509-204728-327-69b211"
    load_settings_onto_instrument(mock_instr_nested, tuid)

    assert mock_instr_nested.a() == 5
    assert mock_instr_nested.mod_a.b() == 3

    assert mock_instr_nested.mod_b.mod_c.c() == 2


def test_load_settings_onto_one_submodule(tmp_test_data_dir, mock_instr_nested):
    """
    Test that we can successfully load the settings of a dummy instrument
    """
    # Always set datadir before instruments
    set_datadir(tmp_test_data_dir)

    # set some random values
    mock_instr_nested.a(23)
    mock_instr_nested.mod_a.b(42)
    mock_instr_nested.mod_b.mod_c.c(23.1)

    # load settings from a dataset
    tuid = "20220509-204728-327-69b211"
    load_settings_onto_instrument(mock_instr_nested.mod_a, tuid)

    assert mock_instr_nested.a() == 23
    assert mock_instr_nested.mod_a.b() == 3
    assert mock_instr_nested.mod_b.mod_c.c() == 23.1


def test_load_settings_onto_one_parameter(tmp_test_data_dir, mock_instr_nested):
    """
    Test that we can successfully load the settings of a dummy instrument
    """
    # Always set datadir before instruments
    set_datadir(tmp_test_data_dir)

    # set some random values
    mock_instr_nested.a(23)
    mock_instr_nested.mod_a.b(42)
    mock_instr_nested.mod_b.mod_c.c(23.1)

    # load settings from a dataset
    tuid = "20220509-204728-327-69b211"
    load_settings_onto_instrument(mock_instr_nested.mod_b.mod_c.c, tuid)

    assert mock_instr_nested.a() == 23
    assert mock_instr_nested.mod_a.b() == 42
    assert mock_instr_nested.mod_b.mod_c.c() == 2


def test_create_plotmon_from_historical(tmp_test_data_dir):
    """
    Test for creating a plotmon based on a provided tuid
    """
    set_datadir(tmp_test_data_dir)

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

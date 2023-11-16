# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name  # in order to keep the fixture in the same file
# pylint: disable=invalid-name

import pytest

import quantify_core.data.handling as dh
from quantify_core.analysis.time_of_flight_analysis import TimeOfFlightAnalysis


@pytest.fixture(
    scope="session",
    params=[
        (
            "20230927-143533-006-fe2167",
            149e-9,  # time of flight
            4.0e-9,  # acquisition delay
        ),
        (
            "20230927-143533-006-fe2167",
            265e-9,  # time of flight
            120e-9,  # acquisition delay
        ),
        (
            "20230927-143514-589-e1a20e-time_of_light_calibration_fit_fail",
            None,  # time of flight
            4.0e-9,  # acquisition delay
        ),
    ],
)
def analysis_and_ref(tmp_test_data_dir, request):
    dh.set_datadir(tmp_test_data_dir)

    tuid, tof, acq_delay = request.param

    analysis = TimeOfFlightAnalysis(tuid=tuid, dataset=dh.load_dataset(tuid)).run(
        acquisition_delay=acq_delay
    )

    return analysis, (tof, acq_delay)


def test_load_fit_results(analysis_and_ref, tmp_test_data_dir):
    dh.set_datadir(tmp_test_data_dir)
    analysis, _ = analysis_and_ref
    for fit_name, fit_result in analysis.fit_results.items():
        loaded_fit_result = TimeOfFlightAnalysis.load_fit_result(
            tuid=analysis.tuid, fit_name=fit_name
        )
        assert loaded_fit_result.params == fit_result.params


# pylint: disable=invalid-name
def test_quantities_of_interest(analysis_and_ref):
    analysis, (tof, acq_delay) = analysis_and_ref
    acq_delay_ns = round(acq_delay * 1e9)

    assert {"fit_success", "fit_msg"} <= set(analysis.quantities_of_interest.keys())
    assert len(set(analysis.quantities_of_interest.keys())) <= 4

    if "tof" in set(analysis.quantities_of_interest.keys()):
        assert {"nco_prop_delay", "tof"} < set(analysis.quantities_of_interest.keys())
        assert analysis.quantities_of_interest["fit_success"]
        assert abs(tof - analysis.quantities_of_interest["tof"]) < 1e-9
    else:
        assert not analysis.quantities_of_interest["fit_success"]
        assert (
            analysis.quantities_of_interest["fit_msg"]
            == "Can not find the Time of flight,\n"
            + f"try to reduce the acquisition_delay (current value: {acq_delay_ns} ns)."
        )

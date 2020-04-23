from quantify.measurement.measurement_control import MeasurementControl


class TestMeasurementControl:

    @classmethod
    def setup_class(cls):
        cls.MC = MeasurementControl(name='MC')

    def test_MeasurementControl_name(self):
        assert self.MC.name == 'MC'

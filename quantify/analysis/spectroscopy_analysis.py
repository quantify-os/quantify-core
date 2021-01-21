import numpy as np
from quantify.analysis import base_analysis as ba
from quantify.analysis import fitting_models as fm


class ResonatorSpectroscopyAnalysis(ba.BaseAnalysis):

    def process_data(self):

        # y0 = amplitude, no check for the amplitude unit as the name/label is often different.
        # y1 = phase in deg, this unit should always be correct
        assert self.dset['y1'].attrs['unit'] == 'deg'

        S21 = self.dset['y0']*np.cos(np.deg2rad(self.dset['y1'])) + \
            1j*self.dset['y0']*np.sin(np.deg2rad(self.dset['y1']))
        self.dset['S21'] = S21
        self.dset['S21'].attrs['name'] = 'S21'
        self.dset['S21'].attrs['unit'] = self.dset['y0'].attrs['unit']
        self.dset['S21'].attrs['long_name'] = 'Transmission, $S_{21}$'

    def run_fitting(self):

        mod = fm.ResonatorModel()

        S21 = np.array(self.dset['S21'])
        f = np.array(self.dset['x0'])
        guess = mod.guess(S21, f=f)
        fit_res = mod.fit(S21, params=guess, f=f)

        self.fit_res = {'hanger_func_complex_SI': fit_res}

    def prepare_figures(self):

        self.plot_dicts = {}

        # iterate over
        self.plot_dicts['S21'] = {
            'plot_fn': ba.plot_basic1D,
            'x': self.dset['x0'].values,
            'xlabel': self.dset['x0'].attrs['long_name'],
            'xunit': self.dset['x0'].attrs['unit'],
            'y': self.dset['S21'].values,
            'ylabel': self.dset['S21'].attrs['long_name'],
            'yunit': self.dset['S21'].attrs['unit'],
            'plot_kw': {"marker": '.', 'label': 'data'},
            'title': 'S21 {}\ntuid: {}'.format(
                self.dset.attrs['name'], self.dset.attrs['tuid'])
        }

        self.plot_dicts['S21-fit'] = {
            'plot_fn': ba.plot_fit,
            'ax_id': 'S21',
            'fit_res': self.fit_res['hanger_func_complex_SI'],
            'plot_init': True}

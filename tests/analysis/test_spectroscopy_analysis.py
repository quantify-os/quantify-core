import quantify.data.handling as dh


from quantify.analysis import spectroscopy_analysis as sa





def test_load_dataset():
    dh.set_datadir(dh._test_dir)

    tuid = '20210118-202044-211-58ddb0'
    a = sa.ResonatorSpectroscopyAnalysis(tuid=tuid)

    # test that the right figures get created.
    assert list(a.figs.keys()) == ['S21']

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_eegfaster
----------------------------------

Tests for `eegfaster` module.
"""

import pytest
import numpy as np
from eegfaster import eegfaster


@pytest.fixture(scope='module')
def data():
    """
    Randomly generates data
    """

    # Some direct accessed for readability
    urnd = np.random.uniform
    nrnd = np.random.normal

    # A controlled experiment will be created with C components containing L
    # time samples and E events.
    C = 100
    L = 1000
    E = 10

    # Time vector
    t = np.linspace(0, 10, E*L)

    # Generator: Original components
    comp_fcn = lambda n: \
        urnd(-1, 1, 1)*np.cos(urnd(0, 1, 1)*n) + \
        urnd(-1, 1, 1)*np.sin(urnd(0, 1, 1)*n)
    # Generator: Added noise
    noise_fcn = lambda n: \
        np.abs(nrnd(1, 0.1, len(n))**4) * \
        np.cos(urnd(0, 0.5, 1)*n)

    # Generate components
    bss_data = []
    for c_n in range(C):
        bss_data.append(comp_fcn(t) + noise_fcn(t))

    # Merge into an array
    bss_data = np.array(bss_data)  # Noisy components (stat point)
    # Reshape components
    bss_data = bss_data.reshape([C, L, E])

    # Generate a mixing matrix
    mix_mat = urnd(-5, 5, [C, C])

    # Generate EOG data
    eog_data = bss_data[:3] + nrnd(0, 1, bss_data[:3].shape)

    # Generate frequency range
    freq_range = [0.1, 0.9]

    # Return only the first 90 components
    return {'bss_data': bss_data[:90], 'mix_mat': mix_mat[:, :90],
            'eog_data': eog_data, 'freq_range': freq_range}


class TestChkParameters:
    """Test _chk_parameters function"""

    def test_mix_mat(self):
        """*mix_mat* must be a 2D array"""
        _test_parameters(eegfaster._chk_parameters, 'mix_mat')
        # Dimensions can also be specified
        eegfaster._chk_parameters(num_mix=4, num_comp=2,
                                  mix_mat=np.zeros([4, 2]))

    def test_bss_data(self):
        """*bss_data* must be a 3-D array"""
        _test_parameters(eegfaster._chk_parameters, 'bss_data')

    def test_mix_bss_congruence(self):
        """*bss_data* must be a 3-D array"""
        _test_parameters(eegfaster._chk_parameters, 'mix_bss')

    def test_eog_data(self):
        """*eog_data* must be a 3-D array"""
        _test_parameters(eegfaster._chk_parameters, 'eog_data')

    def test_eog_bss_congruence(self):
        """*bss_data* must be a 3-D array"""
        _test_parameters(eegfaster._chk_parameters, 'eog_bss')

    def test_freq_range(self):
        """*freq_range* must be a list of length 2"""
        _test_parameters(eegfaster._chk_parameters, 'freq_range')


class TestEOGR:
    """Tests _eog_r function"""

    def test_main(self, data):
        """Functional test"""
        res = eegfaster._eog_r(data['bss_data'], data['eog_data'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self, data):
        """*bss_data* must be a 3-D matrix"""
        _test_parameters(eegfaster._eog_r, 'bss_data',
                         eog_data=data['eog_data'])

    def test_eog_data(self, data):
        """*eog_data* must be a 3-D matrix"""
        _test_parameters(eegfaster._eog_r, 'eog_data',
                         bss_data=data['bss_data'])

    def test_congruence(self, data):
        """
        *bss_data* and *eog_data* must have equal shape for the last 2
        dimensions
        """
        _test_parameters(eegfaster._eog_r, 'eog_bss')

    def test_bss_cdata(self, data):
        """
        *bss_cdata* must return a collapsed version of bss_data
        """
        bss_cdata = []
        eegfaster._eog_r(data['bss_data'], data['eog_data'], bss_cdata)
        assert(isinstance(bss_cdata[0], np.ndarray))
        assert(bss_cdata[0].ndim == 2)
        assert(bss_cdata[0].shape[0] == data['bss_data'].shape[0])
        assert(bss_cdata[0].shape[1] ==
               data['bss_data'].shape[1]*data['bss_data'].shape[2])
        with pytest.raises(TypeError) as err:
            eegfaster._eog_r(data['bss_data'][:, :10], data['eog_data'], 0)
        assert 'bss_cdata must be <' in err.value.message


class TestHurst:
    """Tests _hurst function"""

    def test_main(self, data):
        """Functional test"""
        res = eegfaster._hurst(data['bss_data'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self, data):
        """*bss_data* must be a 3-D matrix"""
        _test_parameters(eegfaster._hurst, 'bss_data')

    def test_cdata(self, data):
        """bss_cdata can also be provided"""
        bss_cdata = []
        eegfaster._eog_r(data['bss_data'], data['eog_data'], bss_cdata)
        res = eegfaster._hurst(bss_cdata=bss_cdata[0])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])
        # bss_cdata must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            eegfaster._hurst(bss_cdata=0)
        assert 'bss_cdata must be <' in err.value.message
        # bss_cdata must be a 2D ndarray
        with pytest.raises(ValueError) as err:
            eegfaster._hurst(bss_cdata=np.zeros([4]*1))
        assert 'bss_cdata must be a 2D array' in err.value.message
        with pytest.raises(ValueError) as err:
            eegfaster._hurst(bss_cdata=np.zeros([4]*3))
        assert 'bss_cdata must be a 2D array' in err.value.message
        # bss_data and bss_cdata can not be simultaneously specified
        with pytest.raises(KeyError) as err:
            eegfaster._hurst(data['bss_data'], bss_cdata[0])
        assert 'Only bss_data or bss_cdata should be' in err.value.message


class TestKurt:
    """Tests _kurt function"""

    def test_main(self, data):
        """Functional test"""
        res = eegfaster._kurt(data['mix_mat'])
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_mix_mat(self):
        """*mix_mat* must be a 2-D matrix"""
        _test_parameters(eegfaster._kurt, 'mix_mat')


class TestdHdf:
    """Tests _dH_df function"""

    def test_main(self, data):
        """Functional test"""
        res = eegfaster._dH_df(data['bss_data'], data['freq_range'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self, data):
        """*bss_data* must be a 3-D matrix"""
        _test_parameters(eegfaster._dH_df, 'bss_data',
                         freq_range=data['freq_range'])

    def test_freq_range(self, data):
        """*freq_range* must be list of floats with length 2"""
        _test_parameters(eegfaster._dH_df, 'freq_range',
                         bss_data=data['bss_data'])


class TestdXdt:
    """Tests _dX_dt function"""

    def test_main(self, data):
        """Functional test"""
        res = eegfaster._dX_dt(data['bss_data'])
        # Result must be a vector of length S
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == data['mix_mat'].shape[1])

    def test_bss_data(self):
        """*bss_data* must be a 3-D matrix"""
        _test_parameters(eegfaster._dX_dt, 'bss_data')


class TestThreshold:
    """Tests _threshold function"""

    def test_main(self):
        """Functional test"""
        c = 100
        res = eegfaster._threshold(np.random.normal(0, 4, c), 3)
        # Result must be a vector of length c
        assert(isinstance(res, np.ndarray))
        assert(res.ndim == 1)
        assert(len(res) == c)

    def test_x(self):
        """x must be a vector"""
        with pytest.raises(TypeError) as err:
            eegfaster._threshold([0, 1], 3)
        assert 'x must be <' in err.value.message

    def test_th(self):
        """th must be a numeric positive value"""
        x = np.random.normal(0, 4, 100)
        with pytest.raises(TypeError) as err:
            eegfaster._threshold(x, [3, 2])
        assert 'th must be <' in err.value.message
        with pytest.raises(ValueError) as err:
            eegfaster._threshold(x, -2)
        assert 'th must be > 0' in err.value.message


class TestArtCmp:
    """Tests eegfaster.art_comp function"""

    def test_main(self, data):
        """Tests the application of faster to data"""
        res = eegfaster.art_comp(data['bss_data'],
                                            data['mix_mat'],
                                            data['eog_data'],
                                            data['freq_range'])
        # This must return 5 results
        assert(isinstance(res, tuple))
        assert(len(res) == 5)

    def test_bss_data(self, data):
        """*bss_data* must be a 3-D matrix"""
        _test_parameters(eegfaster.art_comp, 'bss_data',
                         mix_mat=data['mix_mat'],
                         eog_data=data['eog_data'],
                         freq_range=data['freq_range'])

    def test_eog_data(self, data):
        """*eog_data* must be a 3-D matrix"""
        _test_parameters(eegfaster.art_comp, 'eog_data',
                         mix_mat=data['mix_mat'],
                         bss_data=data['bss_data'],
                         freq_range=data['freq_range'])

    def test_mix_mat(self, data):
        """*mix_mat* must be a 2-D matrix"""
        _test_parameters(eegfaster.art_comp, 'mix_mat',
                         bss_data=data['bss_data'],
                         eog_data=data['eog_data'],
                         freq_range=data['freq_range'])

    def test_freq_range(self, data):
        """*freq_range* must be a list of length 2"""
        _test_parameters(eegfaster.art_comp, 'freq_range',
                         bss_data=data['bss_data'],
                         eog_data=data['eog_data'],
                         mix_mat=data['mix_mat'])


def _test_parameters(fcn, key, *args, **kwargs):
    """
    Tests function parameters.

    This function asserts that a wrongly specified input parameter is detected
    by a function

    Parameters
    ----------
    fcn : functions
        Function to be called
    key : str
        Name of the parameter to be tested
    *args : tuple
        Non-keyword arguments passed to the function, other than "key"
    **kwargs : dict
        Keyword arguments passed to the function, other than "key"
    """

    if key is 'bss_data':
        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data=0., *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data='error', *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(bss_data=[0, 10], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be <' in err.value.message

        # Must be 3D
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*2), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be a 3D array' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*4), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must be a 3D array' in err.value.message

    elif key is 'mix_bss':
        # mix_mat's and bss_data's shapes must be congruent
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*3), mix_mat=np.zeros([4, 2]),
                      *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data must have dimensions MxTxE' in err.value.message

    elif key is 'eog_data':
        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(eog_data=0., *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'eog_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(eog_data='error', *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'eog_data must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(eog_data=[0, 10], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'eog_data must be <' in err.value.message
        # Must be 3D
        with pytest.raises(ValueError) as err:
            res = fcn(eog_data=np.zeros([4]*2), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'eog_data must be a 3D array' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(eog_data=np.zeros([4]*4), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'eog_data must be a 3D array' in err.value.message

    elif key is 'eog_bss':

        # bss_data's and eog_data's shapes must be congruent
        with pytest.raises(ValueError) as err:
            res = fcn(bss_data=np.zeros([4]*3), eog_data=np.zeros([2, 4, 2]),
                      *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'bss_data and eog_data must have equal' in err.value.message

    elif key is 'mix_mat':

        # Must be a numpy.ndarray
        with pytest.raises(TypeError) as err:
            res = fcn(mix_mat=0., *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(mix_mat='error', *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be <' in err.value.message
        # Must be 2D
        with pytest.raises(ValueError) as err:
            res = fcn(mix_mat=np.zeros([2]*1), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be a 2D array' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(mix_mat=np.zeros([2]*3), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'mix_mat must be a 2D array' in err.value.message

    elif key is 'freq_range':
        # Must be a list
        with pytest.raises(TypeError) as err:
            res = fcn(freq_range='error', *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must be <' in err.value.message
        with pytest.raises(TypeError) as err:
            res = fcn(freq_range=np.array([0., 1.]), *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must be <' in err.value.message
        # Must contain float
        with pytest.raises(ValueError) as err:
            res = fcn(freq_range=[0, 1.], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must be a list of <' in err.value.message
        # Must have length 2
        with pytest.raises(ValueError) as err:
            res = fcn(freq_range=[1.], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must have length' in err.value.message
        with pytest.raises(ValueError) as err:
            res = fcn(freq_range=[0., 0.5, 1.], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must have length' in err.value.message
        # Must have values in the range [0, 1]
        with pytest.raises(ValueError) as err:
            res = fcn(freq_range=[0., 1.1], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range values must be in the range' in err.value.message
        # Must have extrictly increasing values
        with pytest.raises(ValueError) as err:
            res = fcn(freq_range=[0.5, 0.1], *args, **kwargs)
            if fcn is eegfaster._chk_parameters and res is not None:
                raise res
        assert 'freq_range must be a strictly increasing' in err.value.message

    else:
        raise SystemError('Test Code Error! '
                          'Specified option {} is not supported.'
                          .format(key))

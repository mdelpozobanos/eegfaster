# -*- coding: utf-8 -*-

"""
===============================
eegfaster.py
===============================

This is the main file from the eegfaster package.
A couple of silly functions and classes are defined here as examples.

"""

import numpy as np
import scipy.stats as sp_stats

# Dimensions are defined here for easy of interpretation of the code.
_ch_dim = 0  # Channel dimension
_c_dim = 0  # Component dimension
_t_dim = 1  # Time dimension
_ev_dim = 2  # Events dimension


def art_comp(bss_data, mix_mat, eog_data, freq_range=None, th=3):
    """
    Automatic classification of EEG components computed by a BSS method.

    This function implements the component classification part of the FASTER
    algorithm described in [Nolan2010]_, which automatically divides the
    components in clean and artifactual groups.

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array with the time course of the BSS component. It must be a 3D array
        with dimensions CxTxE, where C, T and E are the number of components,
        time instants and recorded events respectively.
    mix_mat : numpy.ndarray
        Mixing matrix of the BSS model with dimensions MxC, where M is the
        number of mixed signals and C the number of components.
    eog_data : numpy.ndarray
        Array containing EOG signals, used to compute the correlation of BSS
        components with ocular artifacts. It must be a 3D array with dimensions
        NxTxE, where N, T and E are the number of EOG signals, time instants
        and events respectively. T and E must be the same as in *bss_data*.
    freq_range : list of floats, optional
        List of floats with low and high frequency limits (i.e. [low, high])
        used during the computation of the frequency derivative of the power
        spectrum. Values should be specified in pi-radians and hence limited to
        the range [0, 1].
    th : float, optional
        Threshold applied to the normalized scores

    Returns
    -------
    eog_r_comp : numpy.ndarray
        Logical vector of length C signaling components identified as noise by
        their correlation with EOG channels.
    hurst_comp : numpy.ndarray
        Logical vector of length C signaling components identified as noise by
        their hurst exponent.
    kurt_comp : numpy.ndarray
        Logical vector of length C signaling components identified as noise by
        their spatial kurtosis.
    dH_df_comp : numpy.ndarray
        Logical vector of length C signaling components identified as noise by
        their power spectrum frequency derivative.
    dX_dt_comp : numpy.ndarray
        Logical vector of length C signaling components identified as noise by
        their time derivative.

    #-SPHINX-IGNORE-#
    References
    ----------
    [Nolan2010] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated
    statistical thresholding for eeg artifact rejection. Journal of
    Neuroscience Methods, 192(1):152-162, 2010.
    #-SPHINX-IGNORE-#
    """

    # Compute features
    bss_cdata = []
    eog_r_score = _eog_r(bss_data, eog_data, bss_cdata)
    hurst_score = _hurst(bss_cdata=bss_cdata[0])
    del bss_cdata
    kurt_score = _kurt(mix_mat)
    dh_df_score = _dH_df(bss_data, freq_range)
    dx_dt_score = _dX_dt(bss_data)

    # Normalize and threshold features
    eog_r_comp = _threshold(eog_r_score, th)
    hurst_comp = _threshold(hurst_score, th)
    kurt_comp = _threshold(kurt_score, th)
    dh_dt_comp = _threshold(dh_df_score, th)
    dx_dt_comp = _threshold(dx_dt_score, th)

    # Return results
    return eog_r_comp, hurst_comp, kurt_comp, dh_dt_comp, dx_dt_comp


def _chk_parameters(num_comp=False, num_mix=False, mix_mat=None,
                    bss_data=None, eog_data=None, freq_range=None):
    """
    Checks input parameters.

    Parameters
    ----------
    num_comp : int, optional
        Number of BSS components.
    num_mix : int, optional
        Number of mixed signals.
    mix_mat : numpy.ndarray, optional
        Mixing matrix of the BSS algorithm with dimensions MxC, where M is
        the number of mixed signals and C the number of components.

        .. note:: The length of each dimension dimension can only be checked if
            *mix_mat* and *num_comp* are also specified.

        .. note:: If *mix_mat* and *num_comp* are not specified, the shape of
            *mix_mat* will be used as reference to extract such values.

    bss_data : numpy.ndarray, optional
        3D array with dimensions CxTxE, where C, T and E are the number of
        components, time instants and events respectively.

        .. note:: The length of the first dimension can only be checked if
            *mix_mat* or *num_comp* is also specified.

    eog_data : numpy.ndarray, optional
        3D array dimensions NxTxE, where N, T and E are the number of EOG
        signals, time instants and events respectively. T and E must be the
        same as in *bss_data*.
    freq_range : list
        List of length 2 with strictly increasing floats in the range [0, 1].
    """

    if mix_mat is not None:
        if not isinstance(mix_mat, np.ndarray):
            return TypeError('Parameter mix_mat must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(mix_mat)))
        if mix_mat.ndim != 2:
            return ValueError('Parameter mix_mat must be a 2D array; '
                              'is {}D instead'
                              .format(mix_mat.ndim))
        # Extract/check the number of mixed signals and components
        if num_mix:
            assert(num_mix == mix_mat.shape[0])
        else:
            num_mix = mix_mat.shape[0]
        if num_comp:
            assert(num_comp == mix_mat.shape[1])
        else:
            num_comp = mix_mat.shape[1]

    if bss_data is not None:
        if not isinstance(bss_data, np.ndarray):
            return TypeError('bss_data must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(bss_data)))
        if bss_data.ndim != 3:
            return ValueError('bss_data must be a 3D array; is {}D instead'
                              .format(bss_data.ndim))
        if num_comp and bss_data.shape[0] != num_comp:
            return ValueError('bss_data must have dimensions MxTxE, '
                              'where M, T and E are the number of channels, '
                              'time instants and events respectively; '
                              'bss_data.shape[0] is {} (!= C=={}) instead'
                              .format(bss_data.ndim, num_comp))

    if eog_data is not None:
        if not isinstance(eog_data, np.ndarray):
            return TypeError('eog_data must be {}; is {} instead'
                             .format(type(np.zeros(0)), type(eog_data)))
        if eog_data.ndim != 3:
            return ValueError('eog_data must be a 3D array; is {}D instead'
                              .format(eog_data.ndim))
        if bss_data is not None and not \
                eog_data.shape[1:] == bss_data.shape[1:]:
            return ValueError('bss_data and eog_data must have equal shape on '
                              'the 2nd and 3rd dimensions; they respectivley '
                              'have shapes {} and {} instead'
                              .format(bss_data.shape, eog_data.shape))

    if freq_range is not None:
        if not isinstance(freq_range, list):
            return TypeError('freq_range must be {}; is {} instead'
                             .format(type(list()), type(freq_range)))
        if len(freq_range) != 2:
            return ValueError('freq_range must have length 2; '
                              'it has length {} instead'.
                              format(len(freq_range)))
        for fr in freq_range:
            if not isinstance(fr, float):
                return ValueError('freq_range must be a list of {}; '
                                  'it contains {} instead'
                                  .format(type(0.), type(fr)))
            if fr < 0. or fr > 1.:
                return ValueError(
                    'freq_range values must be in the range [0, 1]'
                )

        if freq_range[0] >= freq_range[1]:
            return ValueError('freq_range must be a strictly increasing list')


def _eog_r(bss_data, eog_data, bss_cdata=None):
    """Computes the correlation between BSS data and EOG channels

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array containing source signals. It must have dimensions CxTxE, where C
        is the number of components, T the number of time instants and E the
        number of events.
    eog_data : numpy.ndarray
        Array with dimensions CxTxE, where C is the number of components, T the
        number of time instants and E the number of events.
    bss_cdata : list
        This is used as an output argument. A version of *bss_data* with time
        and events dimensions collapsed; so that it has shape Cx(T*E), will be
        appended to the list.

        .. Note: This is used internally by faster to avoid recomputing this
            collapsed version for _kurt method.

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """

    def _pearson(x1, x2):
        """
        Computes Pearson's correlation coefficients between x1 and x2 sequences

        Parameters
        ----------
        x1 : numpy.ndarray
            Array with dimensions N1xL, where N1 is the number of sequences and
            L their length.
        x2 : numpy.ndarray
            Array with dimensions N2xL, where N2 is the number of sequences and
            L their length.

        Returns
        -------
        r : numpy.ndarray
            Array with dimensions N1xN2 containing the correlation coefficients
        """
        # Remove mean
        x1 = x1 - x1.mean(axis=1, keepdims=True)
        x2 = x2 - x2.mean(axis=1, keepdims=True)
        # Compute variance
        s1 = (x1**2).sum(axis=1, keepdims=True)
        s2 = (x2**2).sum(axis=1, keepdims=True)
        # Compute r
        return np.dot(x1, x2.T) / np.sqrt(np.dot(s1, s2.T))

    # Collapse time and events in a single dimension
    try:
        xbss_cdata = bss_data.reshape([bss_data.shape[0],
                                       bss_data.shape[1]*bss_data.shape[2]],
                                      order='F')
    except (AttributeError, IndexError, ValueError):
        raise _chk_parameters(bss_data=bss_data)

    try:
        eog_cdata = eog_data.reshape([eog_data.shape[0],
                                      eog_data.shape[1]*eog_data.shape[2]],
                                     order='F')
    except (AttributeError, IndexError, ValueError):
        raise _chk_parameters(eog_data=eog_data)

    # Return bss_cdata if requested
    if bss_cdata is not None:
        try:
            bss_cdata.append(xbss_cdata)
        except AttributeError:
            raise TypeError('bss_cdata must be {}; is {} instead'
                            .format(type(list()), type(bss_cdata)))

    # Return the maximum absolute value of the correlation for each component
    try:
        r = _pearson(xbss_cdata, eog_cdata)
    except ValueError:
        raise _chk_parameters(bss_data=bss_data, eog_data=eog_data)

    return np.abs(r).max(axis=1)


def _hurst(bss_data=None, bss_cdata=None):
    """Computes the Hurst's exponent of BSS components

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array containing source signals. It must have dimensions CxTxE, where C
        is the number of components, T the number of time instants and E the
        number of events.
    bss_cdata : numpy.ndarray
        Array containing source signals with time and events collapsed in a
        single dimension, so that it has shape Cx(T*E). When specified,
        *bss_data* should not be specified.

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """

    # Collapse time and events in a single dimension
    # NOTE: Code planned for array with shame CxTxE
    if bss_cdata is not None and bss_data is not None:
        raise KeyError('Only bss_data or bss_cdata should be specified')
    elif bss_data is not None:
        try:
            bss_cdata = bss_data.reshape([bss_data.shape[0],
                                          bss_data.shape[1]*bss_data.shape[2]],
                                         order='F')
        except (AttributeError, IndexError, ValueError):
            raise _chk_parameters(bss_data=bss_data)

    # Hurst exponent
    # Compute individually for each component
    try:
        num_comp = bss_cdata.shape[0]
    except AttributeError:
        raise TypeError('bss_cdata must be <>; is <> instead'
                        .format(type(np.zeros(0)), type(bss_cdata)))
    h_coef = []
    for c in range(num_comp):
        try:
            h_coef.append(hurst(bss_cdata[c]))
        except TypeError:
            if bss_cdata.ndim is not 2:
                raise ValueError('bss_cdata must be a 2D array; is {}D instead'
                                 .format(bss_cdata.ndim))

    return np.array(h_coef)


def _kurt(mix_mat):
    """Computes the kurtosis of the spatial data

    Parameters
    ----------
    mix_mat : numpy.ndarray
        Mixing matrix of the BSS model with dimensions MxC, where M is the
        number of mixed signals and C the number of components.

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """
    try:
        res = sp_stats.kurtosis(mix_mat, axis=0)
    except IndexError:
        raise _chk_parameters(mix_mat=mix_mat)
    # mix_mat dimensionality has to be checked explicitly, as ND with N != 2
    # does not raise any exception
    if mix_mat.ndim != 2:
        raise _chk_parameters(mix_mat=mix_mat)
    return res


def _dH_df(bss_data, freq_range=None):
    """
    Median frequency derivative value of the PS representation of the BSS
    components

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array containing source signals. It must have dimensions CxTxE, where C
        is the number of components, T the number of time instants and E the
        number of events.
    freq_range : list of floats, optional
        List of floats with low and high frequency limits (i.e. [low, high])
        used during the computation of the frequency derivative of the power
        spectrum. Values should be specified in pi-radians and hence limited to
        the range [0, 1].

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """
    # Frequency derivative of PS. Only check non-filtered frequencies
    try:
        ps = np.fft.fft(bss_data, axis=_t_dim)
    except IndexError:
        raise _chk_parameters(bss_data=bss_data)

    f_dim = _t_dim
    # Keep selected frequency range
    if freq_range is not None:
        # freq_range has to be checked explicitly, as some cases does not raise
        # any exception
        err = _chk_parameters(freq_range=freq_range)
        if err is not None:
            raise err
        # Mask data
        freq = np.linspace(0, 1, ps.shape[f_dim])
        f_mask = np.ones(ps.shape[f_dim], bool)
        f_mask[(freq > freq_range[0]) & (freq < freq_range[1])]
        del freq
    ps = ps.take(np.where(f_mask)[0], axis=f_dim)
    # Compute frequency backward derivative
    ps_df = np.diff(ps, axis=f_dim)
    del ps
    # bss_data dimensionality has to be checked explicitly, as a ND array with
    # N > 3 does not raise any exception. For convenience, check N != 3
    if bss_data.ndim != 3:
        raise _chk_parameters(bss_data=bss_data)
    # Return mean derivative
    return ps_df.mean(axis=(f_dim, _ev_dim))


def _dX_dt(bss_data):
    """
    Median time derivative value of the BSS components

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array containing source signals. It must have dimensions CxTxE, where C
        is the number of components, T the number of time instants and E the
        number of events.

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """

    try:
        # Time backward-derivative of the data
        dt_data = np.diff(bss_data, axis=_t_dim)
        # Collapse time and events in a single dimension
        dt_cdata = dt_data.reshape([dt_data.shape[0],
                                    dt_data.shape[1]*dt_data.shape[2]],
                                   order='F')
    except (IndexError, ValueError):
        raise _chk_parameters(bss_data=bss_data)

    # Return the median derivative value
    return np.median(dt_cdata, axis=1)


def _threshold(x, th):
    """
    Thresholding operator.

    Binarizes the input data *x* setting to 1 values with a z-score greater
    than *th*. The z-score is computed using a trimmed version of the mean and
    standard deviation (sdt). This trim is done rejecting values further than 3
    times the std.

    Parameters
    ----------
    x : numpy.ndarray
        Array with data to be thresholded.
    th : float
        Applied threshold.

    Returns
    -------
    y : numpy.ndarray
        Binarized version of *x*

    """

    # Use z-scores to find outliers
    try:
        z = (x - x.mean()) / x.std()
    except AttributeError:
        raise TypeError('x must be {}; is {} instead'
                        .format(type(np.zeros(0)), type(x)))

    no_outliers = np.abs(z) <= 3
    del z
    # Use mean and std of not-outliers to normalize the data
    n = (x - x[no_outliers].mean()) / x[no_outliers].std()
    # Apply threshold
    try:
        res = n > th
    except:
        if not isinstance(th, (int, long, float)):
            raise TypeError('th must be {} or {}; is {} instead'
                            .format(type(1), type(1.), type(th)))
    # th value must be checked explicitly
    if th <= 0:
        raise ValueError('th must be > 0')
    return res


def hurst(x):
    """FASTER [Nolan2010]_ implementation of the Hurst Exponent.

    Parameters
    ----------
    x : numpy.ndarray
        Vector with the data sequence.

    Returns
    -------
    h : float
        Computed hurst exponent

    #-SPHINX-IGNORE-#
    References
    ----------
    [Nolan2010] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated
    statistical thresholding for eeg artifact rejection. Journal of
    Neuroscience Methods, 192(1):152-162, 2010.
    #-SPHINX-IGNORE-#
    """

    # Get a copy of the data
    x0 = x.copy()
    x0_len = len(x)

    yvals = np.zeros(x0_len)
    xvals = np.zeros(x0_len)
    x1 = np.zeros(x0_len)

    index = 0
    binsize = 1

    while x0_len > 4:

        y = x0.std()
        index += 1
        xvals[index] = binsize
        yvals[index] = binsize*y

        x0_len /= 2
        binsize *= 2
        for ipoints in xrange(x0_len):
            x1[ipoints] = (x0[2*ipoints] + x0[2*ipoints - 1])*0.5

        x0 = x1[:x0_len]

    # First value is always 0
    xvals = xvals[1:index+1]
    yvals = yvals[1:index+1]

    logx = np.log(xvals)
    logy = np.log(yvals)

    p2 = np.polyfit(logx, logy, 1)
    return p2[0]

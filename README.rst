===================================================================
Fully Automatic Statistical Thresholding for EEG artifact Rejection
===================================================================

FASTER is an automatic EEG artifact rejection method based on statistical
thresholding, published by H. Nolan et. al. in 2010 [Nolan2010]_.

The method relies on the dissociation of neural and artifactual activity
through a Blind Source Separation (BSS) algorithm, and the classification of
each extracted component into *clean* or *artifactual*. Components identified
as noisy are then removed from the reconstruction of the EEG.


------------
Code Example
------------

.. note::

  In the current version, only the block in charge of identifying artifactual
  components is available. See :py:func:`eegfaster.eegfaster.art_comp`


------------
Installation
------------

To install this package, you can use the make file. From the root directory of
the package, run:

.. code-block:: bash

  make install

.. note::

  The installation of the dependencies NumPy_ and SciPy_ may fail. It
  is recommended to install these packages manually.

-----
Tests
-----

To test the package against your installed python version, from the root
directory of the package you can run:

.. code-block:: bash

  make test


-------------------
Issues and comments
-------------------

Please, `file an issue`_ if you encounter any problem with the package or if
you have any suggestions.


------------
Contributors
------------

Let people know how they can dive into the project, include important links to
things like issue trackers, irc, twitter accounts if applicable.


==========
References
==========
.. [Nolan2010] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated
    statistical thresholding for eeg artifact rejection. Journal of Neuroscience
    Methods, 192(1):152-162, 2010.


=======
License
=======

The eegfaster framework is open-sourced software licensed
under the `MIT license <http://opensource.org/licenses/MIT>`_.

.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _file an issue: https://github.com/mdelpozobanos/eeglcf/issues

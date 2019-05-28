import pytest
from hmf_emulator import *
import numpy as np
import numpy.testing as npt

#Create an emulator
h = hmf_emulator()

def test_hmf_emulator_load_data():
    npt.assert_equal(h.loaded_data, True)
    attrs = ["data_path", "training_cosmologies",
             "rotation_matrix", "training_data",
             "training_mean", "training_stddev"]
    for attr in attrs:
        npt.assert_equal(hasattr(h, attr), True)
        continue
    return

def test_hmf_emulator_build_emulator():
    npt.assert_equal(h.built, True)
    attrs = ["N_GPs", "GP_list"]
    for attr in attrs:
        npt.assert_equal(hasattr(h, attr), True)
        continue
    npt.assert_equal(h.N_GPs, len(h.GP_list))
    return

def test_hmf_emulator_train_emulator():
    npt.assert_equal(h.trained, True)
    return

def test_hmf_n_in_bins():
    # Cosmology
    cosmology={
        "omega_b": 0.027,
        "omega_cdm": 0.114,
        "w0": -0.82,
        "n_s": 0.975,
        "ln10As": 3.09,
        "H0": 65.,
        "N_eff": 3.
    }
    h.set_cosmology(cosmology)
    #
    ncom = h.n_in_bins((1e12, 1e16), 0.3)
    assert np.isclose(ncom, 0.00480084)

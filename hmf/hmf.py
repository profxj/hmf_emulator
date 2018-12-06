#Test file for extending the Aemulator base class.

from Aemulator import *
import numpy as np
import os, inspect
import scipy.optimize as op
import george
import pickle

class hmf_emulator(Aemulator):

    def __init__(self):
        Aemulator.__init__(self)
        self.loaded_data = False
        self.built       = False
        self.trained     = False
        self.load_data()
        self.build_emulator()
        self.train_emulator()

    def load_data(self, path_to_training_data_directory = None):
        """
        Load training data directly from file, and attach it to this object. 
        This method does not need to be called by the user.
        :param path_to_training_data_directory:
            Location of the training data. Must be in .npy format.
        :return:
            None
        """
        if path_to_training_data_directory is None:
            #Determine the local path to the data files folder
            data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + "/data_files/"
        else:
            data_path = path_to_training_data_directory
        self.data_path = data_path
            
        #Load all training data
        self.training_cosmologies = \
            np.load(data_path+"training_cosmologies.npy")
        self.N_cosmological_params = len(self.training_cosmologies[0])
        self.rotation_matrix      = \
            np.load(data_path+"rotation_matrix.npy")
        self.training_data        = \
            np.load(data_path+"rotated_MF_parameters.npy")
        self.training_mean   = self.training_data[:,:,0] #sample means
        self.training_stddev = self.training_data[:,:,1] #sample stddevs
        self.loaded_data = True
        return

    def build_emulator(self, hyperparams=None):
        """
        Build the emulator directly from loaded training data.
        Optionally provide hyperparameters, 
        if something other than the default is preferred.
        :param hyperparams:
            A dictionary of hyperparameters for the emulator. Default is None.
        :return:
            None
        """
        if not self.loaded_data:
            raise Exception("Need to load training data before building.")

        if hyperparams is None:
            hyperparams = np.std(self.training_cosmologies, 0)

        N_cosmological_params = self.N_cosmological_params
        means  = self.training_mean
        stddev = self.training_stddev

        #Assemble the list of GPs
        self.N_GPs = len(means[0])
        self.GP_list = []
        for i in range(self.N_GPs):
            y    = means[:, i]
            ystd = stddev[:, i]
            kernel = george.kernels.ExpSquaredKernel(hyperparams, ndim=N_cosmological_params)
            gp = george.GP(kernel, mean=np.mean(y),
                           white_noise=np.log(np.mean(ystd)**2))
            gp.compute(self.training_cosmologies, ystd)
            self.GP_list.append(gp)
            continue
        self.built = True
        return

    def train_emulator(self):
        """
        Optimize the hyperparmeters of a built emulator against training data.
        :return:
            None
        """
        if not self.built:
            raise Exception("Need to build before training.")

        means  = self.training_mean

        for i, gp in enumerate(self.GP_list):
            y    = means[:, i]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.lnlikelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_lnlikelihood(y, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
        self.trained = True
        return

    def cache_emulator(self, filename):
        """
        Cache the emulator to a file for easier re-loadig. 
        :param filename:
            The filename where the trained emulator will be cached.
        :return:
            None
        """
        with open(filename, "wb") as output_file:
            pickle.dump(self, output_file)
        return

    def load_emulator(self, filename):
        """
        Load an emulator directly from file, pre-trained.
        :param filename:
            The filename where the trained emulator is located, in a format compatible with
            this object.
        :return:
            None
        """
        if not os.path.isfile(filename):
            raise Exception("%s does not exist to load."%filename)
        with open(filename, "rb") as input_file:
            emu = pickle.load(input_file)
            #Loop over attributes and assign them to this emulator
            for a in dir(emu):
                if not a.startwith("__") and not callable(getattr(emu, a)):
                    setattr(self, a, getattr(emu, a))
                continue
        return

    def predict(self, params):
        """
        Use the emulator to make a prediction at a point in parameter space.
        :param params:
            A dictionary of parameters, where the key is the parameter name and
            value is its value.
        :return:
            pred, the emulator prediction at params. Will be a float or numpy array,
            depending on the quantity being emulated.
        """
        pass

if __name__=="__main__":
    e = hmf_emulator()
    print(e)

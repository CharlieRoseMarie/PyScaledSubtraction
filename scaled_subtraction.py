import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def normalize(v):
    """
    A helper function to normalize a vector. That is, the sum of the square of each component
    equals 1.
    v: A vector to be normalized.
    Returns: The normalized version of the vector.
    Throws an exception if the norm is 0.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        raise Exception("Function can not take a zero vector!")
    return (v/norm)

def create_fitting_func(v1, v2):
    """
    Use this to generate the function to be fitted.
    v1: The first basis vector. In our example, the pure plastic vector.
    v2: The second basis vector. In our example, the pure sugar vector.
    Returns: A function that takes an x value, and two constants which vary with fitting.
    """
    assert(len(v1) == len(v2))

    def func(x, a, b):
        return (a*v1[x] + b*v2[x])
    return func


def calc_scaled_coeff(measured, spectra_one, spectra_two):
    """
    Another helper function. This does the actual work in fitting the data to the model.
    measured: The measured ydata.
    spectra_one: The first basis vector, probably plastic.
    spectra_two: The second basis vector, probably plastic.
    Returns a tuple of the constants calculated.
    """
    xdata = np.arange(len(measured))
    assert(len(xdata) == len(spectra_one) and len(xdata) == len(spectra_two))
    y_func = create_fitting_func(spectra_one, spectra_two)
    popt, pcov = curve_fit(y_func, xdata, measured)
    return popt

def scaled_subtraction(measured, spectra_one, spectra_two):
    """
    The actual scaled_subtraction function intended to be called by other code.
    measured: The measured ydata.
    spectra_one: The first basis vector, probably plastic.
    spectra_two: The second basis vector, probably plastic.
    Returns a tuple of the two calculated vectors. The order of the vectors corrosponds to the order
    that the arguments are given. So if the order is plastic and then sugar, you'll get calculated
    plastic and then calculated sugar.
    """
    spectra_one /= np.linalg.norm(spectra_one)
    spectra_two /= np.linalg.norm(spectra_two)
    coeffs = calc_scaled_coeff(measured, spectra_one, spectra_two)
    loading_one = (measured - coeffs[1]*spectra_two)
    loading_two = (measured - coeffs[0]*spectra_one)
    return (loading_one, loading_two)

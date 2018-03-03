# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:40:32 2016

@author: Erin
"""

from timeout_decorator import timeout, TimeoutError
import numpy as np


class Model():

    likelihood_timeout = 360

    def __init__(self, likelihood, sampled_parameters, l_timeout=None):
        self.likelihood = likelihood
        if l_timeout:
            self.likelihood_timeout = l_timeout
        if type(sampled_parameters) is list:
            self.sampled_parameters = sampled_parameters
        else:
            self.sampled_parameters = [sampled_parameters]

    @timeout(seconds=likelihood_timeout)
    def likelihood_with_timeout(self, q0):
        return self.likelihood(q0)
        
    def total_logp(self, q0):

        prior_logp = 0
        var_start = 0
        for param in self.sampled_parameters:
            var_end = param.dsize + var_start
            try:
                prior_logp += param.prior(q0[var_start:var_end])
            except IndexError:
                #raised if q0 is a single scalar
                prior_logp += param.prior(q0)
            var_start += param.dsize

        try:
            loglike = self.likelihood(q0)
        except TimeoutError:
            loglike = -np.inf
         
        return prior_logp, loglike
    
        
        
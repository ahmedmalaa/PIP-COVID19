
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)


# IF EXISTS REMOVE OS.MKDIR 

from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import pandas as pd
import logging
import traceback
from datetime import date
import pickle
import json
import os
from os import path
import warnings
warnings.filterwarnings('ignore')

today       = date.today()

key_path    = os.getcwd() + "\\" + str(today)
model_path  = os.getcwd() + "\\" + str(today) + "\\" + "models" 
result_path = os.getcwd() + "\\" + str(today) + "\\" + "results" 
projec_path = os.getcwd() + "\\" + str(today) + "\\" + "projections" 

try:
    
    os.mkdir(key_path)
    os.mkdir(model_path)
    os.mkdir(result_path)
    os.mkdir(projec_path)

except OSError:
    
    print ("Creation of the directory %s failed" % model_path)
    
else:
    
    print ("Successfully created the directory %s " % model_path)


from utils.processing import *
from data.DELVE_data import *

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
import ruptures as rpt
from datetime import datetime
#from bayes_opt import BayesianOptimization

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3

from scipy.integrate import odeint

import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  
import ruptures as rpt
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3

from scipy.integrate import odeint

import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import warnings
warnings.filterwarnings('ignore')

"""

def optimize_hyperparameters(N_population, y_train, mobility, val_size=30, 
                             init_points=10, n_iter=15, alpha=1e-1):
        
        
    N_train            = y_train.shape[0]
        
    def evaluate_hyperparameters(**kwargs):
            
        chpts          = int(np.round(kwargs["change_pts"]))
        shift          = int(np.round(kwargs["outbreak_shift"]))
       
        SEIR_model     = SEIR_base(hyperparameters=dict({"outbreak_shift":shift, "change_pts":chpts}), 
                                   N_population=N_population) 
        
        try:
        
            SEIR_model.fit(y_train[:N_train - val_size], mobility[:N_train - val_size, :])
    
            days           = len(y_train) 
            y_pred, _1, _2 = SEIR_model.predict(days)
            y_train_smooth = smooth_curve_1d(y_train)
    
            loss           = -1 * compute_loss(y_train_smooth[N_train - val_size : N_train], 
                                               y_pred[N_train - val_size : N_train])
            
        except:
            
            loss           = -1000000000
            
        return loss

    pbounds   = {"change_pts": (1, 20), "outbreak_shift": (0, 100)}

    optimizer = BayesianOptimization(f=evaluate_hyperparameters,
                                     pbounds=pbounds,
                                     random_state=1,)
    
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter, alpha=alpha)
        
    change_pts     = int(np.round(optimizer.max["params"]["change_pts"]))
    outbreak_shift = int(np.round(optimizer.max["params"]["outbreak_shift"]))
    
    return dict({"outbreak_shift":outbreak_shift, "change_pts":change_pts})

"""

class SEIR_base:

    def __init__(self, N_population, T_incubation=3, T_infectious=9, #T_incubation=1-14, T_infectious=7-12, 
                 T_IC=12, T_CD=7.5, T_CR=6.5, fit_on="deaths",
                 hyperparameters=dict({"outbreak_shift":20, "change_pts":2})):


        """
         Set the model parameters using user input with default values from
         current clinical literature
        """
        
        # Fixed parameters
        
        self.gamma  = 1/T_infectious                      # Infection rate
        self.sigma  = 1/T_incubation                      # Incubation rate
        self.T_IC   = T_IC                                # Number of days from infected to critical
        self.T_CD   = T_CD                                # Number of days from critical to dead
        self.T_CR   = T_CR                                # Number of days from critical to recovered
        self.chpts  = hyperparameters["change_pts"]       # Number of change points in the R0 over time 
        self.shift  = hyperparameters["outbreak_shift"]
        self.fit_on = fit_on
        
        # Clear/Initialize learnable parameters
        
        self.p_IC   = 0   
        self.p_CD   = 0
        self.N      = N_population
    
        
        self.beta_pred   = None
        self.N_train     = 0
        
        self.initialze_hyperparameters()
            
    
    def initialze_hyperparameters(self):
        
        R0_params   = ["R_0_"+str(k) for k in range(self.chpts + 1)] 
        k_params    = ["k_"+str(k) for k in range(self.chpts)] 
        
        R0_dict     =  dict.fromkeys(R0_params)
        k_dict      =  dict.fromkeys(k_params)
        
        self.R0s    = [1.0] * (self.chpts + 1)
        self.k      = [1.0] * (self.chpts)
        
        self.params_init = {"p_IC": (0.05, 0.01, 0.1), 
                            "p_CD": (0.5, 0.05, 0.8)} 
        
        self.params_init.update(R0_dict)
        self.params_init.update(k_dict)
        
        for k in range(len(R0_params)):
            
            self.params_init[R0_params[k]] = (1.0, 0.25, 6.0) 
            
        for m in range(len(k_params)):
            
            self.params_init[k_params[m]]  = (2.5, 0.01, 5.0)
                
        self.params      = dict(self.params_init)

        
    def ODEs(self, compartments, t):
        
        #  Ordinary differential equations modeling the 
        #  compartments of the SEIR model

        
        S, E, I, C, R, D = compartments
        
        mu_IC            = self.p_IC / self.T_IC
        mu_CD            = self.p_CD /self.T_CD  
        
        # FIX THESE
    
        dSdt = -self.beta(t) * I * S / self.N
        dEdt = self.beta(t) * I * S / self.N - self.sigma * E
        dIdt = self.sigma * E - mu_IC * I - self.gamma * (1 - self.p_IC) * I
        dCdt = mu_IC * I - mu_CD * C - (1 - self.p_CD) * (1/self.T_CR) * C
        dRdt = self.gamma * (1 - self.p_IC) * I + (1 - self.p_CD) * (1/self.T_CR) * C
        dDdt = mu_CD * C 
    
        return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
    
    
    def compute_R_0(self, t):
        
        """
          Evaluate the R0 over time
          
        """
        
        Rt_     = []
        
        for k in range(self.chpts):
        
            Rt_curr  = (self.R0s[k] - self.R0s[k+1]) / (1 + np.exp(-1 * self.k[k] * (-t + self.n_bkps[k+1]))) + self.R0s[k+1]
            Rt_.append(Rt_curr)  

        t_mid   = [int((self.n_bkps[k+1] + self.n_bkps[k+2]) / 2) for k in range(self.chpts-1)]
        ret_val = None
        
        for m in range(len(t_mid)):
            
            if (t >= t_mid[m]):
                
                ret_val = Rt_[m + 1]

        if (self.chpts== 1) or (ret_val is None):
            
            ret_val = Rt_[0]
            
        return ret_val

    
    def beta(self, t):
        
        """
          Evaluate the contact rate
          
        """
        
        if (np.floor(t) <= self.N_train) or (self.beta_pred is None):
            
            beta_ = self.compute_R_0(t) * (self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma) / self.sigma
            
        else:
      
            beta_ = self.beta_pred[np.minimum(int(t - self.N_train), len(self.beta_pred) - 1)]  
        
        return beta_

    
    def evaluate(self, days, **kwargs): 
        
        """
          Evaluate the model compartments
          
        """
        
        self.p_CD        = kwargs["p_CD"]
        self.p_IC        = kwargs["p_IC"]
        
        for k in range(len(self.R0s)):
            
            self.R0s[k]  = kwargs["R_0_" + str(k)]
            
        for m in range(len(self.k)):
        
            self.k[m]    = kwargs["k_" + str(m)]
    
        compartments_0   = self.N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0
        t                = np.linspace(0, days - 1, days)
        ret              = odeint(self.ODEs, compartments_0, t)
        S, E, I, C, R, D = ret.T
        R0_t            = [(self.beta(i) * self.sigma) / ((self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma)) for i in range(len(t))]
                
        return t, S, E, I, C, R, D, R0_t
    
    
    def fit(self, deaths_train, cases_train, NPI_train, mobility_train, 
            fit_method="least_squares", smooth=True, n_avg=7): 
        
        lambda_l1       = 3
        L_reg           = 0.01
        
        cases_deaths    = dict({"deaths":np.cumsum(smooth_curve_1d(deaths_train[self.shift:])), 
                                "cases":np.cumsum(smooth_curve_1d(cases_train[self.shift:])),
                                "mix":np.hstack((np.cumsum(smooth_curve_1d(deaths_train[self.shift:])), 
                                                 L_reg * np.cumsum(smooth_curve_1d(cases_train[self.shift:]))))})
        
        NPI_masks       = np.array(NPI_train)[:, 18]
        NPI_non_sd      = np.hstack((np.array([0]), np.diff(NPI_masks)))
        msk_n_bkps      = list(np.where(NPI_non_sd > 0)[0])
  
        cpmodel         = rpt.Pelt(model="rbf").fit(mobility_train[self.shift:, :])
        self.n_bkps     = cpmodel.predict(pen=lambda_l1)
        self.n_bkps     = [0] + list(self.n_bkps) #list(np.sort(np.array(msk_n_bkps + self.n_bkps))) #   
        self.chpts      = len(self.n_bkps) - 2  
 
        self.initialze_hyperparameters()
    
        y_train         = cases_deaths[self.fit_on]
        
        """
          Train the model
          
        """        
        
        self.N_train    = y_train.shape[0] 
         
        mobility_scores = mobility_train[self.shift:, :]
        
        days            = len(deaths_train[self.shift:])
        x_days          = np.linspace(0, days - 1, days, dtype=int)  

        param_          = self.params

        
        def fitter(t, **param_): 
            
            ret = self.evaluate(len(deaths_train[self.shift:]), **param_)
            
            if self.fit_on=="deaths":
                
                out = ret[6][t]
            
            elif self.fit_on=="cases": 
                
                out = ret[3][t]
                
            elif self.fit_on=="mix":
                
                out = np.hstack((ret[6][t], L_reg * ret[3][t]))
    
            return out
        
        mod    = lmfit.Model(fitter)

        for kwarg, (init, mini, maxi) in self.params_init.items():
            
            mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

        params           = mod.make_params()
        train_result     = mod.fit(y_train, params, method=fit_method, t=x_days)
        
        self.best_params = dict(train_result.best_values)
        #self.opt_results = train_result
        
    
    def predict(self, days, R0_forecast=None):
        
        if R0_forecast is None:
            
            self.beta_pred   = None
            
        else:    
        
            self.beta_pred   = R0_forecast * (self.gamma + (self.p_CD/self.T_CD)) * ((self.p_CD/self.T_CD) + self.sigma) / self.sigma
        
        if days > self.shift:
            
            params              = self.best_params
            outputs             = self.evaluate(days - self.shift, **params)
            cases, deaths, R0_t = outputs[3], outputs[6], outputs[7]
        
            deaths              = np.array([0] + list(np.diff(deaths)))
            
            cases               = np.hstack((np.zeros(self.shift), cases))
            deaths              = np.hstack((np.zeros(self.shift), deaths))
            R0_t                = np.hstack((R0_t[0] * np.ones(self.shift), R0_t))
            
        else:    
            
            cases        = np.zeros(days)
            deaths       = np.zeros(days)
            R0_t         = np.zeros(days)
            
        return deaths, cases, R0_t
    


def main(args):


    country_list = ["United Kingdom", "Egypt", "Japan", "Israel", "Croatia",  
                    "United States", "Australia", "Italy", "Germany", "Spain", 
                    "Brazil", "Canada", "Sweden", "Norway",
                    "Finland", "Estonia", "Switzerland", "Ireland", "France",
                    "Netherlands", "Czechia", "Romania", "Belgium", "Algeria",
                    "Argentina", "Chile", "Colombia", "Turkey", "South Africa", 
                    "India", "Pakistan", "Indonesia"]


    country_list = ["United Kingdom", 
                    "United States",  
                    "Italy", 
                    "Germany",  
                    "Brazil", 
                    "Sweden", 
                    "Norway",
                    "France",
                    "Netherlands", 
                    "Belgium"]

    country_dict = get_COVID_DELVE_data(country_list)
    
    N_train      = args.n_train  
    onset_range  = np.linspace(5, 60, 12)
    pred_horizon = 30 * 5

    for country in country_list:
        
        print("\n-----------------------------------------")
        print("    Fitting model for %s" % country)
        print("-----------------------------------------")
        
        deaths_true   = country_dict[country]["Daily deaths"]
        deaths_smooth = smooth_curve_1d(deaths_true)
        cases_true    = country_dict[country]["Daily cases"]
        N_population  = country_dict[country]["Metadata"]["stats_population"] 
        mobility      = country_dict[country]["Smoothened mobility data"]
        NPI_data      = np.array(country_dict[country]["NPI data"])

        deaths_true[deaths_true < 0] = 0
        cases_true[cases_true < 0]   = 0

        onset_loss    = []

        try:

            for onset in onset_range:

                SEIR_model                         = SEIR_base(N_population=N_population, 
                                                               T_infectious=7, T_incubation=8, 
                                                               T_IC=7, T_CD=14, T_CR=14, 
                                                               hyperparameters=dict({"outbreak_shift":int(onset), 
                                                                                     "change_pts":-1})) 
        
                SEIR_model.fit(deaths_true[:N_train], cases_true[:N_train], NPI_data[:N_train, :], mobility[:N_train, :])

                days                               = len(deaths_true) 
                deaths_pred, cases_pred, R0_t_pred = SEIR_model.predict(days)

                onset_loss.append(compute_loss(deaths_smooth[:N_train], deaths_pred[:N_train]))
                
                print("\n * Onset %s --- \n   accuracy %.2f" % (onset, onset_loss[-1]))

            best_onset    = onset_range[np.argmin(np.array(onset_loss))] 
 
            SEIR_model    = SEIR_base(N_population=N_population, 
                                      T_infectious=7, T_incubation=8, 
                                      T_IC=7, T_CD=14, T_CR=14, 
                                      hyperparameters=dict({"outbreak_shift":int(best_onset), 
                                                            "change_pts":-1})) 
        
            SEIR_model.fit(deaths_true[:N_train], cases_true[:N_train], NPI_data[:N_train, :], mobility[:N_train, :])

            days                               = len(deaths_true) + pred_horizon
            deaths_pred, cases_pred, R0_t_pred = SEIR_model.predict(days)

            pickle.dump(SEIR_model, open(model_path + "\\" + country, 'wb'))
            pickle.dump((deaths_pred, R0_t_pred), open(projec_path + "\\" + country, 'wb'))

            plt.figure(figsize=(9, 6))

            x_data        = np.linspace(0, days - 1, days, dtype=int) 
            y_max         = np.maximum(np.max(deaths_true), np.max(deaths_pred)) + 1

            plt.plot(deaths_true, linewidth=1, marker="o", markersize=6, alpha=0.25, color="r", zorder=1)
            plt.plot(x_data[:N_train], deaths_pred[:N_train], color="grey", linewidth=2, zorder=2) 
            plt.plot(x_data[N_train:], deaths_pred[N_train:], color="black", linestyle="--", linewidth=2, zorder=3) 
            plt.vlines(x=[N_train], ymin=0, ymax=y_max, linewidth=1, color="black", zorder=4) 

            plt.plot(x_data[:N_train], deaths_smooth[:N_train], linestyle=":", linewidth=2, color="red", zorder=5)
            plt.plot(x_data[N_train:len(deaths_true)], deaths_smooth[N_train:], linestyle="--", linewidth=2, color="red", zorder=5)

            plt.vlines(x=[N_train + k + 1 for k in range(len(x_data)-N_train)], ymin=0, ymax=y_max, 
                       linewidth=5, color="pink", alpha=0.3, linestyles="--", zorder=0) 

            plt.xlabel("Number of days since the outbreak", labelpad=22)
            plt.ylabel("Daily deaths", labelpad=22)

            plt.xlim(30, days)

            plt.tight_layout()
            plt.savefig(str(today) + "\\" + "results\\" + country + "_projections.pdf", dpi=3000, transparent=True)

        except:

            print("Cannot fit the model for %s" % country)
            traceback.print_exc()    
            


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Fitting global SEIR models")

    parser.add_argument("-n", "--n-train", default=200, type=int)

    args = parser.parse_args()
    
    main(args)


# Conf Interval
# Each country has default policy
# Print numbers on top
# Model update time


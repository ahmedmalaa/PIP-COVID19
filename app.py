
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np
from copy import deepcopy
import time

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from scipy.integrate import odeint

from utils.data_padding import *

torch.manual_seed(1) 


npi_vars       = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                  "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                  "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]

meta_features  = ["stats_population_density", "stats_median_age", "stats_gdp_per_capita", 
                  "stats_smoking", "stats_population_urban", "stats_population_school_age"]


def get_country_features(country_dict_input):
    
    if np.isnan(country_dict_input["Metadata"]["stats_population_school_age"]):
    
        country_dict_input["Metadata"]["stats_population_school_age"] = 15000000
    
    country_dict_input["Metadata"]["stats_population_urban"]      = country_dict_input["Metadata"]["stats_population_urban"]/country_dict_input["Metadata"]["stats_population"]
    country_dict_input["Metadata"]["stats_population_school_age"] = country_dict_input["Metadata"]["stats_population_school_age"]/country_dict_input["Metadata"]["stats_population"]
    
    X_whether = np.array(country_dict_input["wheather data"].fillna(method="ffill"))
    X_cases   = np.array(country_dict_input["Daily cases"] / country_dict_input["Metadata"]["stats_population"]).reshape((-1, 1))
    X_meta    = np.repeat(np.array([country_dict_input["Metadata"][meta_features[k]] for k in range(len(meta_features))]).reshape((1, -1)), X_whether.shape[0], axis=0)
    X_NPI     = np.array(country_dict_input["NPI data"][npi_vars].fillna(method="ffill"))
    X_moblty  = np.array(country_dict_input["Smoothened mobility data"])
    X_strngy  = np.array(country_dict_input["NPI data"]["npi_stringency_index"].fillna(method="ffill").values/100)
    
    return X_whether, X_meta, X_moblty, X_NPI, X_strngy


def get_beta(R0_t_pred, SEIR_model):
    
    beta_t_pred = R0_t_pred * (SEIR_model.gamma + (SEIR_model.p_CD/SEIR_model.T_CD)) * ((SEIR_model.p_CD/SEIR_model.T_CD) + SEIR_model.sigma) / SEIR_model.sigma
    
    return beta_t_pred


def get_R0(beta_t_pred, SEIR_model):
    
    R0_t_pred   = (beta_t_pred * SEIR_model.sigma) / ((SEIR_model.gamma + (SEIR_model.p_CD/SEIR_model.T_CD)) * ((SEIR_model.p_CD/SEIR_model.T_CD) + SEIR_model.sigma)) 
    
    return R0_t_pred


def compute_stringency_index(npi_policy):
    
    """
    
    npi_policy['npi_workplace_closing']              = 3 
    npi_policy['npi_school_closing']                 = 3
    npi_policy['npi_cancel_public_events']           = 2 
    npi_policy['npi_gatherings_restrictions']        = 4
    npi_policy['npi_close_public_transport']         = 2
    npi_policy['npi_stay_at_home']                   = 3
    npi_policy['npi_internal_movement_restrictions'] = 2
    npi_policy['npi_international_travel_controls']  = 4
    
    """
    
    w   = 0.29
    I_1 = (npi_policy["npi_workplace_closing"] * (1-w)/3 + w)
    I_2 = (npi_policy["npi_school_closing"] * (1-w)/3 + w)
    I_3 = (npi_policy["npi_cancel_public_events"] * (1-w)/2 + w)
    I_4 = (npi_policy["npi_gatherings_restrictions"] * (1-w)/4 + w)
    I_5 = (npi_policy["npi_close_public_transport"] * (1-w)/2 + w)
    I_6 = (npi_policy["npi_stay_at_home"] * (1-w)/3 + w)
    I_7 = (npi_policy["npi_internal_movement_restrictions"] * (1-w)/2 + w)
    I_8 = (npi_policy["npi_international_travel_controls"] * (1-w)/4 + w)
    I_9 = 1
    
    I   = (1/9) * (I_1 + I_2 + I_3 + I_4 + I_5 + I_6 + I_7 + I_8 + I_9) 
    
    return I
    
            
def model_loss_single(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.mean(torch.sum(single_loss, axis=0) / torch.sum(masks, axis=0)) 
    
    return loss

def single_losses(model):

    return model.masks * (model(model.X).view(-1, model.MAX_STEPS) - model.y)**2


def model_loss(output, target, masks):
    
    single_loss  = masks * (output - target)**2
    loss         = torch.sum(torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1)))
    
    return loss


def quantile_loss(output, target, masks, q):
    
    single_loss  = masks * ((output - target) * (output >= target)  * q + (target - output) * (output < target)  * (1-q))
    loss         = torch.sum(torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1))) 
    
    return loss    


class R0Forecaster(nn.Module):
    
    def __init__(self, 
                 mode="LSTM",
                 EPOCH=5,
                 BATCH_SIZE=150,
                 MAX_STEPS=50,  
                 INPUT_SIZE=30,     
                 LR=0.01,   
                 OUTPUT_SIZE=1,
                 HIDDEN_UNITS=20,
                 NUM_LAYERS=1,
                 N_STEPS=50,
                 alpha=0.05,
                 beta_max=2,
                 country_parameters=None,
                 country_models=None):
        
        super(R0Forecaster, self).__init__()
        
        self.EPOCH          = EPOCH      
        self.BATCH_SIZE     = BATCH_SIZE
        self.MAX_STEPS      = MAX_STEPS  
        self.INPUT_SIZE     = INPUT_SIZE     
        self.LR             = LR   
        self.OUTPUT_SIZE    = OUTPUT_SIZE
        self.HIDDEN_UNITS   = HIDDEN_UNITS
        self.NUM_LAYERS     = NUM_LAYERS 
        self.N_STEPS        = N_STEPS
        self.q              = alpha
        self.mode           = mode
        self.country_params = country_parameters
        self.country_models = country_models

        rnn_dict = {"RNN" : nn.RNN(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "LSTM": nn.LSTM(input_size = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,),
                    "GRU" : nn.GRU(input_size  = self.INPUT_SIZE, hidden_size = self.HIDDEN_UNITS,  
                                   num_layers  = self.NUM_LAYERS, batch_first = True,)
                    }

        self.rnn            = rnn_dict[self.mode]
        self.out            = nn.Linear(self.HIDDEN_UNITS, 8) 
        self.out_q          = nn.Linear(self.HIDDEN_UNITS, 8) 
        
        self.masks_w        = nn.Parameter(torch.rand(1))        
        self.masks_w_q      = nn.Parameter(torch.rand(1))
        

    def forward(self, x):
        
        if self.mode == "LSTM":

            r_out, (h_n, h_c) = self.rnn(x[:, :, :21], None)   # None represents zero initial hidden state

        else:

            r_out, h_n = self.rnn(x[:, :, :21], None)

        # choose r_out at the last time step

        out   = F.sigmoid(torch.sum(self.out(r_out[:, :, :]) * x[:, :, 21:29], dim=2) - torch.abs(self.masks_w) * (1-x[:, :, 31]) * x[:, :, 30])
        out_q = F.sigmoid(torch.sum(self.out_q(r_out[:, :, :]) * x[:, :, 21:29], dim=2) - torch.abs(self.masks_w_q) * (1-x[:, :, 31]) * x[:, :, 30])
         
        return torch.squeeze(torch.stack([torch.unsqueeze(out, dim=2), torch.unsqueeze(out_q, dim=2)], dim=2), dim=3)
    
    
    def fit(self, X_whether, X_metas, X_mobility, X_NPIs, X_stringency, Y):
        
        self.npi_normalizer   = StandardScaler()
        
        self.model_mob_npi    = self.train_NPI_mobility_layers(X_NPIs, X_mobility, X_metas)
    
        country_names         = list(npi_model.country_params.keys())
        self.beta_nromalizers = dict.fromkeys(list(npi_model.country_params.keys()))
        self.beta_min         = dict.fromkeys(list(npi_model.country_params.keys()))
        
        X_NPI_input           = [np.hstack((X_NPIs[k][:, :], X_stringency[k].reshape((-1, 1)))) for k in range(len(X_NPIs))]
        
        X                     = [np.hstack((np.hstack((np.hstack((X_whether[k], X_metas[k])), X_mobility[k])), X_NPI_input[k])) for k in range(len(X_whether))]
        Y_                    = Y.copy() 
        
        for k in range(len(country_names)):

            self.beta_nromalizers[country_names[k]] = np.max(Y[k] - np.min(Y[k])) 
            self.beta_min[country_names[k]]         = np.min(Y[k])
            Y_[k]                                   = (Y[k] - np.min(Y[k])) / np.max(Y[k] - np.min(Y[k]))
        
        self.normalizer      = StandardScaler()

        self.normalizer.fit(np.array(X).reshape((-1, X[0].shape[1])))
        
        X                    = [self.normalizer.transform(X[k]) for k in range(len(X))]              
        
        for k in range(len(X)):
            
            X[k][:, 31]      = X_stringency[k]
        
        X_padded, _          = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = np.squeeze(padd_arrays(Y_, max_length=self.MAX_STEPS)[0], axis=2), np.squeeze(padd_arrays(Y_, max_length=self.MAX_STEPS)[1], axis=2)
        
        X                    = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y_                   = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks           = Variable(torch.tensor(loss_masks), volatile=True).type(torch.FloatTensor)
        
        self.X               = X
        self.Y               = Y_
        self.masks           = loss_masks
        
        optimizer            = torch.optim.Adam(self.parameters(), lr=self.LR)   # optimize all rnn parameters
        self.loss_func       = quantile_loss 
        
        # training and testing
        for epoch in range(self.EPOCH):

            for step in range(self.N_STEPS):
                
                batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)
                
                x      = torch.tensor(X[batch_indexes, :, :])
                y      = torch.tensor(Y_[batch_indexes])
                msk    = torch.tensor(loss_masks[batch_indexes])
                
                b_x    = Variable(x[:, :, :].view(-1, self.MAX_STEPS, 32))       # self.INPUT_SIZE))   # reshape x to (batch, time_step, input_size)
                b_y    = Variable(y)                                             # batch y
                b_m    = Variable(msk)
                
                output = self(b_x).view(-1, self.MAX_STEPS, 2)                   # rnn output

                L_reg  = 0
                loss   = (1 - L_reg) * model_loss(output[:, :, 0], b_y, b_m) + L_reg * (self.loss_func(output[:, :, 0] + output[:, :, 1], b_y, b_m, self.q) + self.loss_func(output[:, :, 0] - output[:, :, 1], b_y, b_m, 1 - self.q)) 
                
                optimizer.zero_grad()                           # clear gradients for this training step
                loss.backward()                                 # backpropagation, compute gradients
                optimizer.step()                                # apply gradients

                if step % 50 == 0:

                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data)
        
    
    def predict(self, X):
        
        stringency      = []   
        
        for k in range(len(X)):
            
            stringency.append(X[k][:, 31]) 
        
        X               = [self.normalizer.transform(X[k]) for k in range(len(X))] 
        
        for k in range(len(X)):
            
            X[k][:, 31] = stringency[k] 
        
        if type(X) is list:
            
            X_, masks   = padd_arrays(X, max_length=self.MAX_STEPS)
        
        else:
            
            X_, masks   = padd_arrays([X], max_length=self.MAX_STEPS)
        
        X_test          = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)
        
        predicts_      = self(X_test).view(-1, self.MAX_STEPS, 2) 
        prediction_0   = unpadd_arrays(predicts_[:, :, 0].detach().numpy(), masks)
        prediction_1   = unpadd_arrays(predicts_[:, :, 1].detach().numpy(), masks)
        
        return prediction_0, prediction_1
    
    
    def projection(self, days, npi_policy, country="United Kingdom"):
    
        X_whether, X_metas, X_mobility, X_NPIs, X_stringency = get_country_features(self.country_params[country])
        
        X_NPI_input    = np.hstack((X_NPIs, X_stringency.reshape((-1, 1)))) 
        
        X              = [np.hstack((np.hstack((np.hstack((X_whether, X_metas)), X_mobility)), X_NPI_input))]
         
        X_NPI_new      = np.array(list(npi_policy.values()))
        X_NPI_new[-1]  = compute_stringency_index(npi_policy)
        
        X_features     = np.hstack((X_NPI_new[:8].reshape((1, -1)), X_metas[-1, :].reshape((1, -1))))
        X_features     = self.npi_normalizer.transform(X_features)
        
        X_mob_pred     = np.array([self.model_mob_npi[k].predict(X_features) for k in range(len(self.model_mob_npi))])
        
        X_new          = np.hstack((np.hstack((np.hstack((X_whether[-1, :], X_metas[-1, :])), X_mob_pred.reshape((-1,)))), X_NPI_new)) 
                
        X_forecast     = np.vstack((X[0], np.repeat(X_new.reshape((1, -1)), days, axis=0))) 
        
        beta_preds     = self.predict([X_forecast])[0][0] * self.beta_nromalizers[country] + self.beta_min[country]
        beta_pred_CI   = self.predict([X_forecast])[1][0] * self.beta_nromalizers[country] + self.beta_min[country]
        
        beta_pred_u    = beta_preds + beta_pred_CI
        beta_pred_l    = beta_preds - beta_pred_CI
        
        beta_pred_l    = beta_pred_l * (beta_pred_l >= 0) 

        R0_frc         = get_R0(beta_preds, self.country_models[country])
    
        R0_frc_u       = R0_frc + 0.1 #get_R0(beta_pred_u, self.country_models[country])

        R0_frc_l       = R0_frc - 0.1 #get_R0(beta_pred_l, self.country_models[country])
 
        y_pred, _, _   = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc[len(X[0]):])
        y_pred_u, _, _ = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc_u[len(X[0]):])
        y_pred_l, _, _ = self.country_models[country].predict(len(X[0]) + days, R0_forecast=R0_frc_l[len(X[0]):])
    
        return (y_pred, y_pred_u, y_pred_l), (R0_frc, R0_frc_u, R0_frc_l) 
    
    
    def train_NPI_mobility_layers(self, X_NPIs, X_mobility, X_metas):
    
        # add meta vars and normalize

        X_NPI_flat     = np.array(X_NPIs).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
        X_NPI_flat     = X_NPI_flat[:, :8] 
        X_mob_flat     = np.array(X_mobility).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
        X_met_flat     = np.array(X_metas).reshape((X_NPIs[0].shape[0] * len(X_NPIs), -1))
    
        X_features     = np.hstack((X_NPI_flat, X_met_flat))
    
        self.npi_normalizer.fit(X_features)
    
        #model_mob_npi  = [MLPRegressor(hidden_layer_sizes=(500, 500, )) for k in range(X_mob_flat.shape[1])] 
        #model_mob_npi  = [XGBRegressor(n_estimators=100, params={"monotone_constraints": str(tuple([-1] * X_features.shape[1]))}) for k in range(X_mob_flat.shape[1])]
        model_mob_npi  = [LinearRegression() for k in range(X_mob_flat.shape[1])]
        
        for k in range(X_mob_flat.shape[1]):
        
            model_mob_npi[k].fit(self.npi_normalizer.transform(X_features), X_mob_flat[:, k])
    
        return model_mob_npi
        


# ----------------------------------------------------------------------------------------------------------------------------------

# Import required libraries
import os
import pickle
import copy
import datetime as dt
import math

import requests
import pandas as pd
from flask import Flask
import dash
import dash_daq as daq
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_dangerously_set_inner_html
import numpy as np
import os
from os import path

from model.base_model import *
         
external_styles = [
{
    "href": "https://fonts.googleapis.com/css2?family=Open+Sans+Condensed:ital,wght@0,300;0,700;1,300&display=swap",
    "rel": "stylesheet"
},

{
    "href": "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100;300;400;500;700;900&display=swap",
    "rel": "stylesheet"
},

{
    "href": "https://fonts.googleapis.com/css2?family=Ubuntu&display=swap",
    "rel": "stylesheet"
}

]

app               = dash.Dash(__name__, external_stylesheets=external_styles)
server            = app.server
app.title         = "COVID-19 PIP"

POP_UP            = app.get_asset_url("transparent_PIP_logo.png") 

# Define theme color codes

LIGHT_PINK        = "#FF60AA"
DARK_GRAY         = "#323232"
GRAY              = "#808080"
CYAN              = "#95E3FA"

PURPLE_COLOR      = "#AF1CF7"
DARK_PINK         = "#CA1A57"

#-------------------------------------------------------
'''
Helper functions for style formating and data processing

List of helper functions >>
---------------------------
_get_input_HTML_format :: returns cell formating for 
                          html/dcc numerical input 

_get_radioItems_HTML_format :: returns a radio items 
                               list for display
'''
#-------------------------------------------------------

# TO DO: Attack rate plot

'''
COUNTRIES         = ["United States", "United Kingdom", "Italy", "Germany", "Spain", 
                     "Australia", "Brazil", "Canada", "Sweden", "Norway",  "Finland", 
                     "Estonia", "Egypt", "Japan", "Croatia"]
'''

COUNTRIES        = ["United Kingdom"] #["United States", "United Kingdom", "Italy", "Germany", "Brazil", "Japan", "Egypt"]

# load models and data for all countries 

if path.exists(os.getcwd() + "/PIPmodels/global_models"):

  global_models  = pickle.load(open(os.getcwd() + "/PIPmodels/global_models", 'rb'))

else:

  global_models  = dict.fromkeys(COUNTRIES)

  for country in COUNTRIES:

    global_models[country] = pickle.load(open(os.getcwd() + "/2020-09-01/models/" + country, 'rb'))

  pickle.dump(global_models, open(os.getcwd() + "/PIPmodels/global_models", 'wb'))

if path.exists(os.getcwd() + "/PIPmodels/country_data"+"_"+str(dt.date.today())):

  country_data   = pickle.load(open(os.getcwd() + "/PIPmodels/country_data"+"_"+str(dt.date.today()), 'rb'))

else:

  country_data = get_COVID_DELVE_data(COUNTRIES)

  pickle.dump(country_data, open(os.getcwd() + "/PIPmodels/country_data", 'wb'))

 
if path.exists(os.getcwd() + "/PIPmodels/projections"+"_"+str(dt.date.today())):

  global_projections = pickle.load(open(os.getcwd() + "/PIPmodels/projections"+"_"+str(dt.date.today()), 'rb'))

else:

  global_projections = dict.fromkeys(COUNTRIES)

  for country in COUNTRIES:

    global_projections[country] = pickle.load(open(os.getcwd() + "/2020-09-01/projections/" + country, 'rb'))

  pickle.dump(global_models, open(os.getcwd() + "/PIPmodels/global_projections", 'wb'))


npi_model         = pickle.load(open(os.getcwd() + "/PIPmodels/R0Forecaster", 'rb'))

TARGETS           = ["Daily Deaths", "Cumulative Deaths", "Reproduction Number"]

COUNTRY_LIST      = [{'label': COUNTRIES[k], 'value': COUNTRIES[k], "style":{"margin-top":"-.3em", "align": "center"}} for k in range(len(COUNTRIES))]

#TARGET_LIST       = [{'label': TARGETS[k], 'value': k, "style":{"margin-top":"-.3em", "align": "center"}} for k in range(len(TARGETS))]
TARGET_LIST       = [{'label': TARGETS[0], 'value': 0, "style":{"margin-top":"-.3em", "align": "center"}},
                     {'label': TARGETS[2], 'value': 2, "style":{"margin-top":"-.3em", "align": "center"}}]


BOX_SHADOW        = "1px 2px 3px 4px #ccc" 
MARGIN_INPUT      = "20px"
PANEL_COLOR       = "#FBF8F8"

TITLE_STYLE       = {"marginBottom": ".25em", "margin-top": "1em", "margin-left": MARGIN_INPUT, "color":DARK_GRAY, "font-weight": "bold", 
                     "font-size": "12", "font-family": "Noto Sans JP"}
SUBTITLE_STYLE    = {"color":DARK_PINK, "font-size": 13}
SUBTITLE_STYLE_   = {"margin-top":"10px", "color":DARK_PINK, "font-size": 13}
PANEL_TEXT_STYLE  = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "11px", 
                     "font-style": "italic", "font-family":"Noto Sans JP"}
PANEL_TEXT_STYLE2 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "12px", 
                     "font-family":"Noto Sans JP"}
PANEL_TEXT_STYLE3 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "color":GRAY, "font-size": "12px", 
                     "font-family":"Noto Sans JP", "font-weight":"bold"} 
PANEL_TEXT_STYLE4 = {"marginBottom": ".25em", "margin-top": "0em", "margin-left": MARGIN_INPUT, "margin-right": MARGIN_INPUT, "color":GRAY, 
                     "font-size": "12px", "font-family":"Noto Sans JP", "font-weight":"bold"}                     
PANEL_TEXT_STYLE_ = {"marginBottom": "0em", "margin-top": "0em", "color":DARK_GRAY, "font-size": "13px", "font-family":"Open Sans Condensed"}

CAPTION_STYLE     = {"color":"#4E4646", "font-size": 10}
BULLET_STYLE_0    = {"color":"#4E4646", "text-shadow":"#4E4646", "background-color":"#4E4646", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_1    = {"color":"#4F27EC", "text-shadow":"#4F27EC", "background-color":"#4F27EC", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_2    = {"color":"#AF1CF7", "text-shadow":"#AF1CF7", "background-color":"#AF1CF7", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}
BULLET_STYLE_3    = {"color":"#F71C93", "text-shadow":"#F71C93", "background-color":"#F71C93", "border-radius": "10%", "font-size": 10, "width":"7px", "margin-right":"10px"}

name_style        = dict({"color": "#4E4646", 'fontSize': 13, "width": "150px", "marginBottom": ".5em", "textAlign": "left", "font-family": "Noto Sans JP"})
name_style_       = dict({"color": "#4E4646", 'fontSize': 13, "width": "250px", "marginBottom": ".5em", "textAlign": "left", "font-family": "Noto Sans JP"})
input_style       = dict({"width": "100px", "height": "30px", "columnCount": 1, "textAlign": "center", "marginBottom": "1em", "font-size":12, "border-color":LIGHT_PINK})
form_style        = dict({'width' : '10%', 'margin' : '0 auto'})
radio_style       = dict({"width": "150px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11, "border-color":LIGHT_PINK})
radio_style_short = dict({"width": "110px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11})
radio_style_long  = dict({"width": "450px", "color": GRAY, "columnCount": 6, "display": "inline-block", "font-size":11, "font-family": "Noto Sans JP"})
name_style_long   = dict({"color": "#4E4646", 'fontSize': 13, "width": "450px", "columnCount": 3, "marginBottom": ".5em", "textAlign": "left"})
radio_style_her2  = dict({"width": "150px", "color": "#524E4E", "columnCount": 3, "display": "inline-block", "font-size":11})
name_style_her2   = dict({"color": "#4E4646", 'fontSize': 13, "width": "120px", "columnCount": 1, "marginBottom": ".5em", "textAlign": "left"})


npi_variables     = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                     "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                     "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]


def _get_input_HTML_format(name, ID, name_style, input_range, input_step, placeholder, input_style):

    _html_input = html.P(children=[html.Div(name, style=name_style), 
                                      dcc.Input(placeholder=placeholder, type='number', 
                                        min=input_range[0], max=input_range[1], step=input_step, 
                                        style=input_style, id=ID)])

    return _html_input


def _get_radioItems_HTML_format(name, ID, name_style, options, radio_style):

    _html_radioItem = html.P(children=[html.Div(name, style=name_style), 
                                       dcc.RadioItems(options=options, value=1, style=radio_style, id=ID)])

    return _html_radioItem


def _get_toggle_switch(name, name_style, color_style, ID):

    _html_toggle    = html.P(children=[html.Div(name, style=name_style),
                                       daq.ToggleSwitch(color=color_style, size=30, value=True,  
                                                        label=['No', 'Yes'], style={"font-size":9, "font-family": "Noto Sans JP", "color":GRAY}, id=ID)], 
                                                        style={"width": "100px", "font-size":9})

    return _html_toggle

def HORIZONTAL_SPACE(space_size):
    
    return dbc.Row(dbc.Col(html.Div(" ", style={"marginBottom": str(space_size) + "em"})))

def VERTICAL_SPACE(space_size):

    return dbc.Col(html.Div(" "), style={"width": str(space_size) + "px"})

#-------------------------------------------------------
'''
App layout components

List of layout components >>
---------------------------
HEADER :: Logo display and navigation buttons on the app
          header area 

PATIENT_INFO_FORM :: form that reads patient information
                     to compute and display risk
'''
#-------------------------------------------------------


# Create the **header** with logo and navigation buttons
#-------------------------------------------------------

LEARN_BUTTON    = html.A(dbc.Button("Learn More", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/", className="two columns")
WEBSITE_BUTTON  = html.A(dbc.Button("Go back to website", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/", className="two columns")
FEEDBACK_BUTTON = html.A(dbc.Button("Send Feedback", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/contact-us/", className="two columns")
GITHUB_BUTTON   = html.A(dbc.Button("GitHub", style={"bgcolor": "gray"}), href="https://www.vanderschaar-lab.com/contact-us/", className="two columns")
UPDATE_BUTTON   = dbc.Button("Update Projections", style={"bgcolor": "gray"})


HEADER  = html.Div([

      html.Div(
        [ 
        
        dbc.Row([dbc.Col(html.Img(src=app.get_asset_url("logo.png"), id="adjutorium-logo", style={"height": "100px", 'textAlign': 'left',
                                                                                                  "width": "auto",})),
                 VERTICAL_SPACE(325),   
                 dbc.Col(LEARN_BUTTON),
                 VERTICAL_SPACE(20),
                 dbc.Col(WEBSITE_BUTTON),
                 VERTICAL_SPACE(20),
                 dbc.Col(FEEDBACK_BUTTON), 
                 VERTICAL_SPACE(20),
                 dbc.Col(GITHUB_BUTTON)]),         
       
        ], style={"margin-left":"5ex"}, className="header"),

    ],

)


# Create the *Patient Information form* for app body
# -------------------------------------------------- 

# Input, name & HTML form styling dictionaries


COUNTRY_DROPMENU  = dcc.Dropdown(id='country', options= COUNTRY_LIST, value="United Kingdom",  
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

REGION_DROPMENU   = dcc.Dropdown(id='region', options= COUNTRY_LIST, disabled=True, 
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

TARGET_DROPMENU   = dcc.Dropdown(id='target', options= TARGET_LIST, value=0, 
                                 placeholder="  ", style={"width":"150px", "height": "30px", "font-size": 11, "border-color":GRAY, "color":GRAY,
                                                          "font-color":GRAY, "margin-top":"-.1em", "textAlign": "left", "font-family": "Noto Sans JP", 
                                                          "vertical-align":"top", "display": "inline-block"}) 

HORIZON_SLIDER    = dcc.Slider(id='horizonslider', marks={7: "1w", 30: "1m", 60: "2m", 90: "3m"}, min=7, 
                               max=90, value=30, step=1, updatemode="drag", tooltip={"always_visible":False})


MASK_SLIDER       = dcc.RadioItems(id='maskslider',
                                   options=[{'label': 'No policy measures', 'value': 0}, 
                                            {'label': 'Recommended', 'value': 1},
                                            {'label': 'Limited mandate', 'value': 2},
                                            {'label': 'Universal', 'value': 3}], value=1, 
                                   labelStyle={"display": "inline-block", "font-size": 11,
                                               "font-family": "Noto Sans JP", "color":GRAY, "width":"50%"},
                                   inputStyle={"color":CYAN}) 


SOCIAL_DIST_OPT   = dcc.Checklist(id='socialdistance',
                                  options=[{'label': 'Workplace closure', 'value': 0},
                                           {'label': 'Public events cancellation', 'value': 1},
                                           {'label': 'Public transport closure', 'value': 2},
                                           {'label': 'Gatherings restrictions', 'value': 3},
                                           {'label': 'Shelter-in-place' , 'value': 4},
                                           {'label': 'Internal movement restrictions' , 'value': 5},
                                           {'label': 'Travel restrictions' , 'value': 6}],
                                  value=[0],
                                  labelStyle={"display": "inline-block", "font-size": 11,
                                              "font-family": "Noto Sans JP", "color":GRAY, "width":"50%"}) 


DISPLAY_LIST_2    = dcc.Checklist(options=[{'label': 'Show PIP model fit', 'value': 1}],
                                  labelStyle={"font-size": 11, "font-family": "Noto Sans JP", "color":GRAY, 'display': 'inline-block'},
                                  id="pipfit") 


DISPLAY_LIST_3    = dcc.Checklist(options=[{'label': 'Show confidence intervals', 'value': 1}], 
                                  value=[1],
                                  labelStyle={"font-size": 11, "font-family": "Noto Sans JP", "color":GRAY, 'display': 'inline-block'},
                                  id="confidenceint") 


Num_days          = (dt.date.today() - dt.date(2020, 1, 1)).days 
BEGIN_DATE        = dcc.Slider(id='dateslider', marks={0: "Jan 1st, 2020", Num_days: "Today"}, 
                               min=0, max=Num_days, value=0, step=1, updatemode="drag", tooltip={"always_visible":False})

HORIZON_NOTE      = "*w = week, m = month." 
REQUEST_NOTE      = "Select a geographical location and the required forecast." 
REQUEST_NOTE_2    = "Select the non-pharmaceutical interventions (NPIs) to be applied in the geographical area selected above." 

COUNTRY_SELECT    = html.P(children=[html.Div("Country", style=name_style), COUNTRY_DROPMENU])
REGION_SELECT     = html.P(children=[html.Div("Region", style=name_style), REGION_DROPMENU])
TARGET_SELECT     = html.P(children=[html.Div("Forecast Target", style=name_style), TARGET_DROPMENU])
HORIZON_SELECT    = html.P(children=[html.Div("Forecast Days*", style=name_style), HORIZON_SLIDER])
MASK_SELECT       = html.P(children=[html.Div("Mask Policy", style=name_style), MASK_SLIDER])
SOCIAL_SELECT     = html.P(children=[html.Div("Social Distancing Measures", style=name_style_), SOCIAL_DIST_OPT])
BEGIN_SELECT      = html.P(children=[html.Div("View from", style=PANEL_TEXT_STYLE4), BEGIN_DATE]) 
SCHOOL_CLOSURE    = _get_toggle_switch(name="School Closure ", name_style=name_style, color_style=CYAN, ID="school_closure")


PATIENT_INFO_FORM = html.Div(
    [

      html.Div(
        [ 
        
        dbc.Row(dbc.Col(html.Div("Forecast Settings", style={"marginBottom": "0.5em", "margin-top": "1em", "margin-left": MARGIN_INPUT, 
                                                               "color":DARK_GRAY, "font-weight": "bold", "font-size": "11", "font-family": 'Noto Sans JP'}))),
        dbc.Row(dbc.Col(html.Div(REQUEST_NOTE, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [
                dbc.Col(COUNTRY_SELECT), 
                VERTICAL_SPACE(40),
                dbc.Col(REGION_SELECT),  
                
            ], style={"margin-left": "40px"}
               ),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [   dbc.Col(TARGET_SELECT),
                VERTICAL_SPACE(40),
                dbc.Col(HORIZON_SELECT), 
            ], style={"margin-left": "40px"}
               ),
        HORIZONTAL_SPACE(.5),
        dbc.Row([VERTICAL_SPACE(200), dbc.Col(html.Div(HORIZON_NOTE, style=PANEL_TEXT_STYLE))]),
        HORIZONTAL_SPACE(1),

        ],  style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "450px"}), 

    html.Div(
        [     
        dbc.Row(dbc.Col(html.Div("Policy Scenario", style={"marginBottom": "1em", "margin-top": "1em", "margin-left": MARGIN_INPUT,
                                                                    "color":DARK_GRAY, "font-weight": "bold", "font-size": "11", "font-family":'Noto Sans JP'}))), 
        dbc.Row(dbc.Col(html.Div(REQUEST_NOTE_2, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(1),
        dbc.Row(
            [   
                VERTICAL_SPACE(40),
                dbc.Col(SCHOOL_CLOSURE), 
                VERTICAL_SPACE(60),
                dbc.Col(MASK_SELECT),
            ], style={"margin-left": MARGIN_INPUT}
               ),
        HORIZONTAL_SPACE(.5),
        dbc.Row(
            [   
                VERTICAL_SPACE(25),
                dbc.Col(SOCIAL_SELECT), 
            ], style={"margin-left": MARGIN_INPUT}
               ),
        #HORIZONTAL_SPACE(1),
        #dbc.Row(
        #    [   
        #        VERTICAL_SPACE(100),
        #        dbc.Col(UPDATE_BUTTON),
        #    ], style={"margin-left": MARGIN_INPUT}
        #       ),
        HORIZONTAL_SPACE(1.5),
        ], style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "450px"}),

    ],

)

# Create the results display panel

CAUTION_STATEMENT = "Disclaimer: PIP uses machine learning to predict the most likely trajectory of COVID-19 deaths based on current knowledge and data, but will not provide 100% accurate predictions. Click on the 'Learn more' button to read our model's assumptions and limitations."


RESULTS_DISPLAY   = html.Div(
    [

      html.Div( 
        [  

        dbc.Row(dbc.Col(html.Div("COVID-19 Forecasts", style=TITLE_STYLE))),
        dbc.Row(dbc.Col(html.Div(CAUTION_STATEMENT, style=PANEL_TEXT_STYLE2))),
        HORIZONTAL_SPACE(.5),
        dbc.Row([dbc.Col(html.Div("Display Options", style=PANEL_TEXT_STYLE3)), VERTICAL_SPACE(10), DISPLAY_LIST_2,
                 VERTICAL_SPACE(10), DISPLAY_LIST_3, VERTICAL_SPACE(50), BEGIN_SELECT]), 
        HORIZONTAL_SPACE(2),
        dbc.Row(html.Div(dcc.Graph(id="covid_19_forecasts", config={'displayModeBar': False}), style={"marginBottom": ".5em", "margin-top": "0em", "margin-left": MARGIN_INPUT})),                          
        HORIZONTAL_SPACE(1.25),
        ],  style={"box-shadow": BOX_SHADOW, "margin": MARGIN_INPUT, "background-color": PANEL_COLOR, "width": "800px"}),

    ]

)

#-----------------------------------------------------
'''
APP Layout: contains the app header, the information
form and the displayed graphs
'''
#-----------------------------------------------------

          #<h2 class="modal__title" id="modal-1-title">
          #  Micromodal
          #</h2>

popup = html.Div([
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
         <div class="modal micromodal-slide" id="modal-1" aria-hidden="true">
    <div class="modal__overlay" tabindex="-1" data-micromodal-close>
      <div class="modal__container" role="dialog" aria-modal="true" aria-labelledby="modal-1-title">
        <header class="modal__header">
          <div class="image-wrapper">
            <img src="assets/transparent_PIP_logo.png" style="width:100%;" alt="image">
          </div>
          <button class="modal__close" aria-label="Close modal" data-micromodal-close></button>
        </header>
        <main class="modal__content" id="modal-1-content">
          <p>
            PIP is an online tool that uses machine learning to predict the impact of non-pharmaceutical policy measures on the future trajectory of COVID-19 deaths. The model is designed and trained based on current knowledge and data, and does not provide 100% accurate predictions. Please make sure to discuss the projections of PIP with your local health officials and experts. Visit our <a href="https://www.vanderschaar-lab.com/policy-impact-predictor-for-covid-19/"> website </a> to learn more about our model's assumptions and limitations.
          </p>
        </main>
        <footer class="modal__footer">
          <button class="modal__btn" style="font-size:12px" data-micromodal-close aria-label="Close this dialog window">Start using PIP now!</button>
        </footer>
      </div>
    </div>
  </div>
    '''),
]) 

app.layout = html.Div([popup, HEADER, html.Div([PATIENT_INFO_FORM, RESULTS_DISPLAY], className="row app-center")])
 
@app.callback(
    Output("covid_19_forecasts", "figure"),
    [Input("target", "value"), Input("horizonslider", "value"), Input("maskslider", "value"), Input("country", "value"),
     Input("pipfit", "value"), Input("confidenceint", "value"), Input("dateslider", "value"), Input("socialdistance", "value"),
     Input("school_closure", "value")]) 


def update_risk_score(target, horizonslider, maskslider, country, pipfit, confidenceint, dateslider, socialdistance, school_closure):


    """
    Set X and Y axes based on input callbacks             

    """

    SHOW_PIP_FIT      = False
    SHOW_CONFIDENCE   = True

    if type(pipfit)==list:

      if len(pipfit) > 0:

        SHOW_PIP_FIT  = True

    if type(confidenceint) !=list or len(confidenceint)==0:   
      
      SHOW_CONFIDENCE = False    

    Y_AXIS_NAME       = TARGETS[target] 
    TODAY_DATE        = dt.datetime.today()
    BEGIN_YEAR        = dt.datetime(2020, 1, 1)
    DAYS_TILL_TODAY   = (TODAY_DATE - BEGIN_YEAR).days
    END_DATE          = TODAY_DATE + dt.timedelta(days=horizonslider)
    START_DATE        = BEGIN_YEAR + dt.timedelta(days=dateslider)
    DATE_RANGE        = pd.date_range(start=START_DATE, end=END_DATE) 
    TOTAL_NUM_DAYS    = len(DATE_RANGE)
    TRUE_DEATHS_DATES = pd.date_range(start=START_DATE, end=TODAY_DATE)
    FORECAST_DATES    = pd.date_range(start=TODAY_DATE + dt.timedelta(days=1), end=END_DATE)
    MAX_HORIZON       = 120 
    PLOT_RATIO        = 0.1

    predictive_model  = global_models[country]
    country_DELVE_dat = country_data[country]
    deaths_true       = country_DELVE_dat["Daily deaths"]
    NPI_data          = country_data[country]["NPI data"]

    deaths_true[deaths_true < 0] = 0
    deaths_smooth                = smooth_curve_1d(deaths_true)
    cumulative_deaths            = np.cumsum(deaths_true)

    if maskslider==0:

      deaths_pred, _, R_t = predictive_model.predict(DAYS_TILL_TODAY + MAX_HORIZON, R0_forecast=1*np.ones(MAX_HORIZON))
      deaths_forecast     = deaths_pred[DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
      PIP_MODEL_FIT       = deaths_pred[:DAYS_TILL_TODAY-1] 

    else:
      
      deaths_forecast     = global_projections[country][0][DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1]
      PIP_MODEL_FIT       = global_projections[country][0][:DAYS_TILL_TODAY-1] 
     
    R0_t_forecast         = global_projections[country][3][dateslider:DAYS_TILL_TODAY + horizonslider-1]
    deaths_CI_l           = global_projections[country][2][:horizonslider]
    deaths_CI_u           = global_projections[country][1][:horizonslider] 

    deaths_CI             = 50 * np.ones(len(deaths_CI_u))

    # -------------------------------------------------------------------------------------------------------------------------------------------

    """
    Compute projections

    """

    npi_vars              = ["npi_workplace_closing", "npi_school_closing", "npi_cancel_public_events",  
                             "npi_gatherings_restrictions", "npi_close_public_transport", "npi_stay_at_home", 
                             "npi_internal_movement_restrictions", "npi_international_travel_controls", "npi_masks"]

    npi_policy                                       = dict.fromkeys(npi_vars)                         

    npi_policy['npi_workplace_closing']              = (np.sum(np.array(socialdistance)==0) > 0) * 3           #3 
    npi_policy['npi_school_closing']                 = ((school_closure==True) | (school_closure=="Yes")) * 2 + 1  #3
    npi_policy['npi_cancel_public_events']           = (np.sum(np.array(socialdistance)==1) > 0) * 2           #2 
    npi_policy['npi_gatherings_restrictions']        = (np.sum(np.array(socialdistance)==3) > 0) * 1 + 3       #4
    npi_policy['npi_close_public_transport']         = (np.sum(np.array(socialdistance)==2) > 0) * 2           #2
    npi_policy['npi_stay_at_home']                   = (np.sum(np.array(socialdistance)==4) ==0) * 3           #3
    npi_policy['npi_internal_movement_restrictions'] = (np.sum(np.array(socialdistance)==2) > 0) * 1 + 1       #2
    npi_policy['npi_international_travel_controls']  = (np.sum(np.array(socialdistance)==2) > 0) * 4           #4  

    npi_policy['npi_masks']                          = maskslider                                              #3
    npi_policy['stringency']                         = 0   


    (y_pred, y_pred_u, y_pred_l), (R0_frc, R0_frc_u, R0_frc_l) = npi_model.projection(days=MAX_HORIZON, npi_policy=npi_policy, country=country)

    deaths_forecast       = y_pred[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]
    R0_t_forecast         = smooth_curve_1d(R0_frc[dateslider : DAYS_TILL_TODAY + horizonslider - 1])
    cum_death_forecast    = np.cumsum(deaths_forecast) + np.sum(deaths_true)
    deaths_forecast_u     = y_pred_u[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]
    deaths_forecast_l     = y_pred_l[DAYS_TILL_TODAY - 1 : DAYS_TILL_TODAY + horizonslider - 1]

    # -------------------------------------------------------------------------------------------------------------------------------------------

    if target==0:

      Y_MAX_VAL           = np.maximum(np.max(deaths_smooth), np.max(y_pred[:DAYS_TILL_TODAY]))
      Y_MAX_VAL           = Y_MAX_VAL * (1 + PLOT_RATIO)

    elif target==1:
      
      Y_MAX_VAL           = np.max(cum_death_forecast) + np.max(deaths_CI_u) 
      Y_MAX_VAL           = Y_MAX_VAL * (1 + PLOT_RATIO)

    elif target==2:
      
      Y_MAX_VAL           = 6 

    LINE_WIDTH        = 2
    LINE_WIDTH_       = 3
    _OPACITY_1        = 0.2
    _OPACITY_2        = 0.3
    _OPACITY_3        = 0.4
    COLOR_1           = ("#4F27EC", "rgba(79, 39, 236, " + str(_OPACITY_1)+")")
    COLOR_2           = ("#AF1CF7", "rgba(175, 28, 247, " + str(_OPACITY_2)+")")
    COLOR_3           = ("#F5B7B1", "rgba(245, 183, 177, " + str(_OPACITY_3)+")")   

    LINE_STYLE_0      = {"color":"#2C2B2D", "width":LINE_WIDTH, "dash": "dot"}
    LINE_STYLE_1      = {"color":COLOR_1[0], "width":LINE_WIDTH}
    LINE_STYLE_2      = {"color":COLOR_2[0], "width":LINE_WIDTH}
    LINE_STYLE_3      = {"color":COLOR_3[0], "width":LINE_WIDTH}
    LINE_STYLE_4      = {"color":GRAY, "width":LINE_WIDTH, "dash": "dot"}

    TRUE_DEATH_STYLE  = {"color":"red", "opacity":.25}
    PIP_FIT_STYLE     = {"color":PURPLE_COLOR, "symbol":"cross", "opacity":.5}
    SMTH_DEATH_STYLE  = {"color":"red", "width":LINE_WIDTH, "dash": "dot"}
    R0_STYLE          = {"color":PURPLE_COLOR, "width":LINE_WIDTH_}
    R0_STYLE_         = {"color":PURPLE_COLOR, "width":LINE_WIDTH_, "dash": "dot"}
    FORECAST_STYLE    = {"color":"black", "width":LINE_WIDTH_, "dash": "dot"}
    FORECAST_STYLE_   = {"color":COLOR_3[1], "width":LINE_WIDTH}

    pip_fit_dict      = {"x":TRUE_DEATHS_DATES, "y": PIP_MODEL_FIT[dateslider:], "mode":"markers", "marker":PIP_FIT_STYLE, 
                         "name": "PIP Model Fit"}

    deaths_true_dict  = {"x":TRUE_DEATHS_DATES, "y": deaths_true[dateslider:], "mode":"markers", "marker":TRUE_DEATH_STYLE, 
                         "name": "Daily Deaths"}

    death_smooth_dict = {"x":TRUE_DEATHS_DATES, "y": deaths_smooth[dateslider:], "mode":"lines", "line":SMTH_DEATH_STYLE, 
                         "name": "7-day Average Deaths"} 

    death_frcst_dict  = {"x":FORECAST_DATES, "y": deaths_forecast, "mode":"lines", "line":FORECAST_STYLE, 
                         "name": "Deaths Forecast"} 

    death_frcst_dictu = {"x":FORECAST_DATES, "y": deaths_forecast_u, "mode":"lines", "line":FORECAST_STYLE_, 
                         "fill":"tonextx", "fillcolor":COLOR_3[1], "name": "Deaths Forecast  (Upper)"}  

    death_frcst_dictl = {"x":FORECAST_DATES, "y": deaths_forecast_l, "mode":"lines", "line":FORECAST_STYLE_, 
                         "name": "Deaths Forecast (Lower)"}  

    cum_frcst_dictu   = {"x":FORECAST_DATES, "y": np.sum(deaths_true) + np.cumsum(deaths_CI_u), "mode":"lines", "line":FORECAST_STYLE_, 
                         "fill":"tonexty", "fillcolor":COLOR_3[1], "name": "Cumulative Deaths (Upper)"}  

    cum_frcst_dictl   = {"x":FORECAST_DATES, "y": np.sum(deaths_true) + np.cumsum(deaths_CI_l), "mode":"lines", "line":FORECAST_STYLE_, 
                         "name": "Cumulative Deaths (Lower)"}                                                                

    cum_frcst_dict    = {"x":FORECAST_DATES, "y": cum_death_forecast, "mode":"lines", "line":FORECAST_STYLE, 
                         "name": "Cumulative Deaths Forecast"}                                           

    cum_death_dict    = {"x":TRUE_DEATHS_DATES, "y": cumulative_deaths[dateslider:], "mode":"lines", "line":SMTH_DEATH_STYLE, 
                         "name": "Cumulative Deaths"} 

    R0_frcst_dict     = {"x":TRUE_DEATHS_DATES, "y": R0_t_forecast[dateslider:DAYS_TILL_TODAY], "mode":"lines", "line":R0_STYLE, 
                         "name": "R0_t"} 

    R0_pred_dict      = {"x":FORECAST_DATES, "y": R0_t_forecast[DAYS_TILL_TODAY-1:DAYS_TILL_TODAY + horizonslider-1], "mode":"lines", "line":R0_STYLE_, 
                         "name": "R0_t"}                      

    today_line        = {"x":[dt.date.today() for k in range(int(Y_MAX_VAL))], "y": np.linspace(0, int(Y_MAX_VAL), int(Y_MAX_VAL)), "mode":"lines", "line":LINE_STYLE_4, 
                         "name": "Forecast day"}                         

    if target==0:

      DATA_DICT       = [deaths_true_dict, death_smooth_dict, death_frcst_dict]

      if SHOW_CONFIDENCE:

        DATA_DICT     = DATA_DICT + [death_frcst_dictl, death_frcst_dictu]

      if SHOW_PIP_FIT:

        DATA_DICT     = DATA_DICT + [pip_fit_dict]

      DATA_DICT       = DATA_DICT + [today_line]  

    elif target==1:
      
      DATA_DICT       = [cum_death_dict, cum_frcst_dict]

      if SHOW_CONFIDENCE:

        DATA_DICT     = DATA_DICT + [cum_frcst_dictu, cum_frcst_dictl]

      DATA_DICT       = DATA_DICT + [today_line]  

    elif target==2:

      DATA_DICT       = [R0_frcst_dict, R0_pred_dict, today_line]

    plot_dict = {
        "data": DATA_DICT,
        "showlegend": False, 
        "layout": {
            "legend":{"x":-10, "y":0, "bgcolor": "rgba(0,0,0,0)", "font-size":8},
            "showlegend": False, 
            "font-size":11,
            "width":775,
            "height":383,
            "plot_bgcolor":PANEL_COLOR,
            "paper_bgcolor":PANEL_COLOR,
            "margin":dict(l=60, r=50, t=30, b=40),
            "fill":"toself", "fillcolor":"violet",
            "title":"<b> Confirmed deaths: </b>" + " "+ str(format(int(np.sum(deaths_true)), ",")) + " (as of today)  | <b> Projected total deaths: </b>" + " "+ str(format(int(np.ceil(cum_death_forecast[-1])), ",")) + "  by  " + END_DATE.strftime("%b %d, %Y"), 
            "titlefont":dict(size=13, color=GRAY, family="Noto Sans JP"),
            "xaxis":go.layout.XAxis(title_text="<b> Date </b>", type="date", tickvals=DATE_RANGE, dtick=10, tickmode="auto", 
                                    zeroline=False, titlefont=dict(size=12, color=GRAY, family="Noto Sans JP")),
            "yaxis":go.layout.YAxis(title_text="<b> " + Y_AXIS_NAME + " </b>", tickmode="auto", range=[0, Y_MAX_VAL], 
                                    titlefont=dict(size=12, color=GRAY, family="Noto Sans JP"))} 
    }

    return plot_dict


@app.callback(
    Output("socialdistance", "value"), 
    [Input("country", "value")]) 


def update_NPIs(country):

  social_dist_measure = ["npi_workplace_closing", "npi_cancel_public_events", "npi_close_public_transport", 
                         "npi_gatherings_restrictions", "npi_stay_at_home", "npi_internal_movement_restrictions", 
                         "npi_international_travel_controls"]

  country_NPI_data    = country_data[country]["NPI data"][social_dist_measure].fillna(method="ffill")
  NPI_selections      = np.where(np.array(country_NPI_data)[-1, :]>0)[0]

  return list(NPI_selections)


@app.callback(
    Output("maskslider", "value"), 
    [Input("country", "value")]) 

def update_mask_info(country):

  country_NPI_data    = country_data[country]["NPI data"]["npi_masks"].fillna(method="ffill")
  mask_selection      = int(np.array(country_NPI_data)[-1])

  return mask_selection  


@app.callback(
    Output("school_closure", "value"), 
    [Input("country", "value")]) 

def update_school_info(country):

  school_options      = ["No", "Yes"]
  country_NPI_data    = country_data[country]["NPI data"]["npi_school_closing"].fillna(method="ffill")
  school_closure      = school_options [(np.array(country_NPI_data)[-1] > 0) * 1]

  return school_closure    


#-----------------------------------------------------
'''
Main

'''
#-----------------------------------------------------
if __name__ == '__main__':
    
    app.server.run(debug=True, threaded=True)


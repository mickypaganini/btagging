# -*- coding: utf-8 -*-
'''
Info:
    Combine output of MV2 with output of either IPMP or IP3D
Author: 
    Michela Paganini - Yale/CERN
    michela.paganini@cern.ch
To-do:
    Balance flavor fractions.
'''

import pandas as pd
import pandautils as pup
import numpy as np
import os
import logging
import deepdish.io as io
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from rootpy.vector import LorentzVector, Vector3
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Highway, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import cPickle
from viz import add_curve, calculate_roc, ROC_plotter

# -- custom utility functions defined in this folder
from utils import configure_logging
from data_utils import replaceInfNaN, apply_calojet_cuts, reweight_to_l

def main(iptagger, root_paths, model_id):
    configure_logging()
    logger = logging.getLogger("Combine_MV2IP")
    logger.info("Running on: {}".format(iptagger))

    branches, training_vars = set_features(iptagger)
    logger.info('Creating dataframe...')
    df = pup.root2panda('../data/final_production/*', 
        'bTag_AntiKt4EMTopoJets', 
        branches = branches)

    logger.info('Transforming variables...')
    df = transformVars(df, iptagger)

    logger.info('Flattening df...')
    df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in df.iterkv()})
    del df

    logger.info('Applying cuts...')
    df_flat = apply_calojet_cuts(df_flat)

    logger.info('Will train on {}'. format(training_vars))
    logger.info('Creating X, y, w, mv2c10...')
    y = df_flat['jet_LabDr_HadF'].values
    mv2c10 = df_flat['jet_mv2c10'].values
    # -- slice df by only keeping the training variables
    X = df_flat[training_vars].values
    pteta = df_flat[['jet_pt', 'abs(jet_eta)']].values
    #w = reweight_to_b(pteta, y, pt_col=0, eta_col=1)
    w = reweight_to_l(pteta, y, pt_col=0, eta_col=1)
    del df_flat, pteta

    logger.info('Shuffling, splitting, scaling...')
    ix = np.array(range(len(y)))
    X_train, X_test, y_train, y_test, w_train, w_test, \
    ix_train, ix_test, mv2c10_train, mv2c10_test = train_test_split(
        X, y, w, ix, mv2c10, train_size=0.6
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    le = LabelEncoder()
    net = Sequential()
    net.add(Dense(16, input_shape=(X_train.shape[1], ), activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(4, activation='softmax'))
    net.summary()
    net.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    weights_path = iptagger + '-' + model_id + '-progress.h5'
    try:
        logger.info('Trying to load weights from ' + weights_path)
        net.load_weights(weights_path)
        logger.info('Weights found and loaded from ' + weights_path)
    except IOError:
        logger.info('No weights found in ' + weights_path)

    # -- train 
    try:
        net.fit(
            X_train, le.fit_transform(y_train),
            verbose=True, 
            batch_size=64, 
            sample_weight=w_train,
            callbacks = [
                EarlyStopping(verbose=True, patience=100, monitor='val_loss'),
                ModelCheckpoint(weights_path, monitor='val_loss', verbose=True, save_best_only=True)
            ],
            nb_epoch=200, 
            validation_split=0.3
        ) 
    except KeyboardInterrupt:
        print '\n Stopping early.'

    # -- load in best network
    net.load_weights(weights_path)

    # -- test
    print 'Testing...'
    yhat = net.predict(X_test, verbose=True) 

    # -- save the predicions to numpy file
    np.save('yhat-{}-{}.npy'.format(iptagger, model_id), yhat)
    test = {
        'X' : X_test,
        'y' : y_test,
        'w' : w_test,
        'mv2c10' : mv2c10_test
    }
    # -- plot performance
    performance(yhat, test, iptagger)

# ----------------------------------------------------------------- 

def performance(yhat, test, iptagger):
    # -- Find flavors after applying cuts:
    bl_sel = (test['y'] == 5) | (test['y'] == 0)
    cl_sel = (test['y'] == 4) | (test['y'] == 0)
    bc_sel = (test['y'] == 5) | (test['y'] == 4)

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 0])[bl_sel])
    bl_curves = {}
    add_curve(r'MV2c10+' + iptagger, 'green', 
          calculate_roc( test['y'][bl_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 0])[bl_sel][fin1]),
          bl_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( test['y'][bl_sel] == 5, test['mv2c10'][bl_sel]),
          bl_curves)
    cPickle.dump(bl_curves, open('ROC_MV2c10+' + iptagger + '_' + model_id + '_bl.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bl_curves, title=r'DL1 + IP Taggers Combination', min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 10000)
    fg.savefig('ROC_MV2c10+' + iptagger + '_' + model_id + '_bl.pdf')

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 1])[bc_sel])
    bc_curves = {}
    add_curve(r'MV2c10+' + iptagger, 'green', 
          calculate_roc( test['y'][bc_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 1])[bc_sel][fin1]),
          bc_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( test['y'][bc_sel] == 5, test['mv2c10'][bc_sel]),
          bc_curves)
    cPickle.dump(bc_curves, open('ROC_MV2c10+' + iptagger + '_' + model_id + '_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bc_curves, title=r'DL1 + IP Taggers Combination', min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 100)
    fg.savefig('ROC_MV2c10+' + iptagger + '_' + model_id + '_bc.pdf')

# ----------------------------------------------------------------- 

def transformVars(df, iptagger):
    '''
    modifies the variables to create the ones that mv2 uses
    inserts default values when needed
    saves new variables in the dataframe
    Args:
    -----
        df: pandas dataframe containing all the interesting variables as extracted from the .root file
        iptagger: string, either 'ip3d' or 'ipmp'
    Returns:
    --------
        modified mv2-compliant dataframe
    '''
    # -- modify features and set default values
    df['abs(jet_eta)'] = abs(df['jet_eta'])

    # -- create new IPxD features
    if iptagger == 'ip3d':
        df['jet_ip3d_pu'].apply(lambda x: replaceInfNaN(x, -20))
        df['jet_ip3d_pb'].apply(lambda x: replaceInfNaN(x, -20))
        df['jet_ip3d_pc'].apply(lambda x: replaceInfNaN(x, -20))
    elif iptagger == 'ipmp':
        df['jet_ipmp_pu'].apply(lambda x: replaceInfNaN(x, -20))
        df['jet_ipmp_pb'].apply(lambda x: replaceInfNaN(x, -20))
        df['jet_ipmp_pc'].apply(lambda x: replaceInfNaN(x, -20))
        df['jet_ipmp_ptau'].apply(lambda x: replaceInfNaN(x, -20))
    else:
        raise ValueError('iptagger can only be ip3d or ipmp')

    return df

# ----------------------------------------------------------------- 

def set_features(iptagger):
    '''
    List of useful branches to extract from the file.
    Depending on whether iptagger is ipmp or ip3d, different branches will be activated.
    Note:
        Raises ValueError if the name of the tagger is not in ['ipmp', 'ip3d']
    '''
    # -- generically useful variables (for cuts, etc.)
    branches = [
        'jet_pt',
        'jet_eta',
        'jet_aliveAfterOR',
        'jet_aliveAfterORmu',
        'jet_nConst',
        'jet_LabDr_HadF',
        'jet_mv2c10'
        ]
    # -- variables to be combined with a NN
    training_vars = [
        'jet_mv2c10'
        ]

    if iptagger == 'ip3d':
        branches += [
            'jet_ip3d_pu', 'jet_ip3d_pc', 'jet_ip3d_pb'
        ]
        training_vars += [
            'jet_ip3d_pu', 'jet_ip3d_pc', 'jet_ip3d_pb'
        ]
    elif iptagger == 'ipmp':
        branches += [
            'jet_ipmp_pu', 'jet_ipmp_pc', 'jet_ipmp_pb', 'jet_ipmp_ptau'
            ]
        training_vars += [
            'jet_ipmp_pu', 'jet_ipmp_pc', 'jet_ipmp_pb', 'jet_ipmp_ptau'
            ]
    else:
        raise ValueError('iptagger can only be ip3d or ipmp')
    return branches, training_vars

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import sys
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    # Select whether you want to combine MV2 with the outputs of IP3D or IPMP
    parser.add_argument('iptagger', help="select 'ip3d' or 'ipmp'")
    # Give a name to your model for future retrieval
    parser.add_argument('model_id', help="token to identify the model")
    # Path to the root files
    parser.add_argument('input', type=str, nargs="+", help="Path to root files, e.g. /path/to/pattern*.root")
    args = parser.parse_args()

    sys.exit(main(args.iptagger, args.input, args.model_id))


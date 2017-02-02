import deepdish.io as io
from rootpy.vector import LorentzVector, Vector3
import matplotlib.pyplot as plt
import pandas as pd
import pandautils as pup
import numpy as np
import math
import os
import sys
#from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

'''
------
TO DO:
 i.   consider removing taus from the training (instead of just assigning a weight of 0 but keeping them as a class)
 ii.  try to speed up transformVars
 iii. write checks for transformVars to make sure output doesn't change
 iv.  LabelEncoder should not be needed -- remove it?
------
'''

def main(inputfiles, treename='bTag_AntiKt2PV0TrackJets'):

    configure_logging()
    logger = logging.getLogger('ProcessTrackJetData')

    # -- import root files into df
    logger.info('Importing ROOT files into pandas dataframes')
    df = pup.root2panda(inputfiles, treename, branches = [
            'jet_pt', 'jet_eta','jet_phi', 'jet_m', 'jet_ip2d_pu', 
            'jet_ip2d_pc', 'jet_ip2d_pb', 'jet_ip3d_pu', 'jet_ip3d_pc','jet_ip3d_pb',
            'jet_sv1_vtx_x', 'jet_sv1_vtx_y', 'jet_sv1_vtx_z', 'jet_sv1_ntrkv',
            'jet_sv1_m','jet_sv1_efc','jet_sv1_n2t','jet_sv1_sig3d',
            'jet_jf_n2t','jet_jf_ntrkAtVx','jet_jf_nvtx','jet_jf_nvtx1t','jet_jf_m',
            'jet_jf_efc','jet_jf_sig3d', 'jet_jf_deta', 'jet_jf_dphi', 'PVx', 'PVy', 'PVz',
            'jet_aliveAfterOR', 'jet_aliveAfterORmu', 'jet_nConst', 'jet_LabDr_HadF'])

    # -- Insert default values, calculate MV2 variables from the branches in df
    logger.info('Creating MV2 variables')
    df = transformVars(df)

    # -- Flatten from event-flat to jet-flat
    # -- Before doing so, remove event-level variables such as PVx,y,z
    logger.info('Flattening dataframe')
    df.drop(['PVx', 'PVy', 'PVz'], axis=1, inplace=True)
    df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in df.iterkv()})

    # -- apply eta, pt, OR cuts from b-tagging recommendations
    logger.info('Applying cuts')
    df_flat = applycuts(df_flat)

    # -- build X, y, w
    # -- target values
    y = df_flat['jet_LabDr_HadF'].values

    # -- slice df by only keeping the 24 variables for MV2 training
    training_vars = [
        'jet_pt', 
        'abs(jet_eta)', 
        'jet_ip2',
        'jet_ip2_c',
        'jet_ip2_cu',
        'jet_ip3',
        'jet_ip3_c',
        'jet_ip3_cu',
        'jet_sv1_ntrkv',
        'jet_sv1_m',
        'jet_sv1_efc',
        'jet_sv1_n2t',
        'jet_sv1_Lxy',
        'jet_sv1_L3d',
        'jet_sv1_sig3d',
        'jet_sv1_dR',
        'jet_jf_n2t',
        'jet_jf_ntrkAtVx',
        'jet_jf_nvtx',
        'jet_jf_nvtx1t',
        'jet_jf_m',
        'jet_jf_efc',
        'jet_jf_dR',
        'jet_jf_sig3d'] 
    X = df_flat[training_vars].as_matrix()
    logger.info('2D pT and eta reweighting of charm and light to bottom distribution')
    w = reweight_to_b(X, y)

    X, y, w = remove_tau(X, y, w)

    # -- turn classes 0, 4, 5, 15 to 0, 1, 2, 3
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    # -- randomly shuffle and split into train and test set
    logger.info('Shuffling and splitting')
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, train_size = 0.6)

    # -- save out to hdf5
    logger.info('Saving data to hdf5')
    io.save(open('train_data.h5', 'wb'), {'X' : X_train, 'y' : y_train, 'w' : w_train})
    io.save(open('test_data.h5', 'wb'), {'X' : X_test, 'y' : y_test, 'w' : w_test})

# -----------------------------------------------------------------

def configure_logging():
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))

# -----------------------------------------------------------------

def _replaceInfNaN(x, value):
    '''
    function to replace Inf and NaN with a default value
    Args:
    -----
        x:     arr of values that might be Inf or NaN
        value: default value to replace Inf or Nan with
    Returns:
    --------
        x:     same as input x, but with Inf or Nan raplaced by value
    '''
    x[np.isfinite( x ) == False] = value 
    return x

# -----------------------------------------------------------------

def match_shape(arr, ref):
    '''
    function to replace 1d array into array of arrays to match event-jets structure
    Args:
    -----
        arr: jet-flat array 
        ref: array of arrays (event-jet structure) as reference for shape matching
    Returns:
    --------
        the initial arr but with the same event-jet structure as ref
    Raises:
    -------
        ValueError
    '''
    shape = [len(a) for a in ref]
    if len(arr) != np.sum(shape):
        raise ValueError('Incompatible shapes: len(arr) = {}, total elements in ref: {}'.format(len(arr), np.sum(shape)))
  
    return [arr[ptr:(ptr + nobj)].tolist() for (ptr, nobj) in zip(np.cumsum([0] + shape[:-1]), shape)]

# ----------------------------------------------------------------- 

def transformVars(df):
    '''
    modifies the variables to create the ones that mv2 uses, inserts default values when needed, saves new variables
    in the dataframe
    Args:
    -----
        df: pandas dataframe containing all the interesting variables as extracted from the .root file
    Returns:
    --------
        modified mv2-compliant dataframe
    '''
    # -- modify features and set default values
    for (pu,pb,pc) in zip(df['jet_ip2d_pu'],df['jet_ip2d_pb'],df['jet_ip2d_pc']) :
        pu[np.logical_or(pu >= 10, pu <-1)] = -1
        pb[np.logical_or(pu >= 10, pu <-1)] = -1
        pc[np.logical_or(pu >= 10, pu <-1)] = -1
        
    for (pu,pb,pc) in zip(df['jet_ip3d_pu'],df['jet_ip3d_pb'],df['jet_ip3d_pc']) :
        pu[pu >= 10] = -1
        pb[pu >= 10] = -1
        pc[pu >= 10] = -1

    # -- create new IPxD features
    df['abs(jet_eta)'] = abs(df['jet_eta'])
    df['jet_ip2'] = (df['jet_ip2d_pb'] / df['jet_ip2d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip2_c'] = (df['jet_ip2d_pb'] / df['jet_ip2d_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip2_cu'] = (df['jet_ip2d_pc'] / df['jet_ip2d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip3'] = (df['jet_ip3d_pb'] / df['jet_ip3d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip3_c'] = (df['jet_ip3d_pb'] / df['jet_ip3d_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip3_cu'] = (df['jet_ip3d_pc'] / df['jet_ip3d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))

    # -- SV1 features
    dx = df['jet_sv1_vtx_x']-df['PVx']
    dy = df['jet_sv1_vtx_y']-df['PVy']
    dz = df['jet_sv1_vtx_z']-df['PVz']

    v_jet = LorentzVector()
    pv2sv = Vector3()
    sv1_L3d = []
    sv1_Lxy = []
    dR = [] 

    for index, dxi in enumerate(dx): # loop thru events
        sv1_L3d_ev = []
        sv1L_ev = []
        dR_ev = []
        for jet in xrange(len(dxi)): # loop thru jets
            v_jet.SetPtEtaPhiM(df['jet_pt'][index][jet], df['jet_eta'][index][jet], df['jet_phi'][index][jet], df['jet_m'][index][jet])
            if (dxi[jet].size != 0):
                sv1_L3d_ev.append(np.sqrt(pow(dx[index][jet], 2) + pow(dy[index][jet], 2) + pow(dz[index][jet], 2))[0])
                sv1L_ev.append(math.hypot(dx[index][jet], dy[index][jet]))
                
                pv2sv.SetXYZ(dx[index][jet], dy[index][jet], dz[index][jet])
                jetAxis = Vector3(v_jet.Px(), v_jet.Py(), v_jet.Pz())
                dR_ev.append(pv2sv.DeltaR(jetAxis))
            else: 
                dR_ev.append(-1)   
                sv1L_ev.append(-100)
                sv1_L3d_ev.append(-100)
             
        sv1_Lxy.append(sv1L_ev)
        dR.append(dR_ev) 
        sv1_L3d.append(sv1_L3d_ev)
        
    df['jet_sv1_dR'] = dR 
    df['jet_sv1_Lxy'] = sv1_Lxy
    df['jet_sv1_L3d'] = sv1_L3d

    # -- add more default values for sv1 variables
    sv1_vtx_ok = match_shape(np.asarray([len(el) for event in df['jet_sv1_vtx_x'] for el in event]), df['jet_pt'])

    for (ok4event, sv1_ntkv4event, sv1_n2t4event, sv1_mass4event, sv1_efrc4event, sv1_sig34event) in zip(sv1_vtx_ok, df['jet_sv1_ntrkv'], df['jet_sv1_n2t'], df['jet_sv1_m'], df['jet_sv1_efc'], df['jet_sv1_sig3d']): 
        sv1_ntkv4event[np.asarray(ok4event) == 0] = -1
        sv1_n2t4event[np.asarray(ok4event) == 0] = -1 
        sv1_mass4event[np.asarray(ok4event) == 0] = -1000
        sv1_efrc4event[np.asarray(ok4event) == 0] = -1 
        sv1_sig34event[np.asarray(ok4event) == 0] = -100

    # -- JF features
    jf_dR = []
    for eventN, (etas, phis, masses) in enumerate(zip(df['jet_jf_deta'], df['jet_jf_dphi'], df['jet_jf_m'])): # loop thru events
        jf_dR_ev = []
        for m in xrange(len(masses)): # loop thru jets
            if (masses[m] > 0):
                jf_dR_ev.append(np.sqrt(etas[m] * etas[m] + phis[m] * phis[m]))
            else:
                jf_dR_ev.append(-10)
        jf_dR.append(jf_dR_ev)
    df['jet_jf_dR'] = jf_dR

    # -- add more default values for jf variables
    for (jf_mass,jf_n2tv,jf_ntrkv,jf_nvtx,jf_nvtx1t,jf_efrc,jf_sig3) in zip(df['jet_jf_m'],df['jet_jf_n2t'],df['jet_jf_ntrkAtVx'],df['jet_jf_nvtx'],df['jet_jf_nvtx1t'],df['jet_jf_efc'],df['jet_jf_sig3d']):
        jf_n2tv[jf_mass <= 0] = -1;
        jf_ntrkv[jf_mass <= 0] = -1;
        jf_nvtx[jf_mass <= 0]  = -1;
        jf_nvtx1t[jf_mass <= 0]= -1;
        jf_mass[jf_mass <= 0]  = -1e3;
        jf_efrc[jf_mass <= 0]  = -1;
        jf_sig3[jf_mass <= 0]  = -100;

    return df

# ----------------------------------------------------------------- 

def applycuts(df):

    cuts = (abs(df['jet_eta']) < 2.5) & \
           (df['jet_pt'] > 10e3) & \
           (df['jet_aliveAfterOR'] == 1) & \
           (df['jet_aliveAfterORmu'] == 1) & \
           (df['jet_nConst'] > 1)

    df = df[cuts].reset_index(drop=True)
    return df

# ----------------------------------------------------------------- 

def reweight_to_b(X, y):
    '''
    Definition:
    -----------
        pT and eta reweighting to the b-distributions
    Returns:
    --------
        w: array of weights

    '''

    pt_bins = [10, 50, 100, 150, 200, 300, 500, max(X[:, 0])]
    eta_bins = np.linspace(0, 2.5, max(X[:, 1]))

    b_bins = plt.hist2d(X[y == 5, 0]/1000, X[y == 5, 1], bins=[pt_bins, eta_bins])
    c_bins = plt.hist2d(X[y == 4, 0]/1000, X[y == 4, 1], bins=[pt_bins, eta_bins])
    l_bins = plt.hist2d(X[y == 0, 0]/1000, X[y == 0, 1], bins=[pt_bins, eta_bins])

    wb = np.ones(X[y == 5].shape[0])

    wc = [(b_bins[0]/c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, 0]/1000, b_bins[1]) - 1, 
        np.digitize(X[y == 4, 1], b_bins[2]) - 1
        )]

    wl = [(b_bins[0]/l_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 0, 0]/1000, b_bins[1]) - 1, 
        np.digitize(X[y == 0, 1], b_bins[2]) - 1
        )]

    w = np.zeros(len(y))
    w[y==5] = wb 
    w[y==4] = wc
    w[y==0] = wl

    return w

# ----------------------------------------------------------------- 

def remove_tau(X, y, w):

    X = X[y != 15]
    y = y[y != 15]
    w = w[y != 15]

    return X, y, w

# ----------------------------------------------------------------- 

if __name__ == '__main__':

    # -- read in arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfiles", help="input .root file name")
    parser.add_argument("--tree", help="jet collection (tree) name")
    args = parser.parse_args()

    # -- pass arguments to main
    if (args.tree != None):
        sys.exit(main(args.inputfiles, args.tree))

    else:
        sys.exit(main(args.inputfiles))

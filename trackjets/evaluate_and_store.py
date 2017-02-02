#!/usr/bin/env python

from root_numpy.tmva import add_classification_events, evaluate_reader
from rootpy.io import root_open
from rootpy import stl
from rootpy.tree import Tree
from rootpy.vector import Vector3, LorentzVector
from ROOT import TMVA, TFile
import xml.etree.ElementTree as ET
from array import array
import pandautils as pup
import numpy as np
import math
import os
import sys

def main(weights, picklename, filename, treename = 'bTag_AntiKt2PV0TrackJets'):
    '''
    evaluate the tmva method after transforming input data into right format
    Args:
    -----
        weights:    .xml file out of mv2 training containing bdt parameters
        picklename: name of the output pickle to store new mv2 values
        filename:   .root file with ntuples used to evaluate the tmva method
        treename:   (optional) name of the TTree to consider 
    Returns:
    --------
        status
    Raises:
    -------
        nothing yet, but to be improved
    '''
    print 'Parsing XML file...'
    # -- Load XML file
    tree = ET.parse(weights) 
    root = tree.getroot()

    # -- Get list of variable names from XML file
    var_list = [var.attrib['Label'] for var in root.findall('Variables')[0].findall('Variable')]

    # -- Count the input variables that go into MV2:
    n_vars = len(var_list)
    
    
    print 'Loading .root file for evaluation...'
    # -- Get ntuples:
    df = pup.root2panda(filename, treename, branches = ['jet_pt', 'jet_eta','jet_phi', 'jet_m', 'jet_ip2d_pu', 
        'jet_ip2d_pc', 'jet_ip2d_pb', 'jet_ip3d_pu', 'jet_ip3d_pc','jet_ip3d_pb',
        'jet_sv1_vtx_x', 'jet_sv1_vtx_y', 'jet_sv1_vtx_z', 'jet_sv1_ntrkv',
        'jet_sv1_m','jet_sv1_efc','jet_sv1_n2t','jet_sv1_sig3d',
        'jet_jf_n2t','jet_jf_ntrkAtVx','jet_jf_nvtx','jet_jf_nvtx1t','jet_jf_m',
        'jet_jf_efc','jet_jf_sig3d', 'jet_jf_deta', 'jet_jf_dphi', 'PVx', 'PVy', 'PVz' ])

    # -- Insert default values, calculate MV2 variables from the branches in df
    df = transformVars(df)

    # -- Map ntuple names to var_list
    names_mapping = {
        'pt':'jet_pt',
        'abs(eta)':'abs(jet_eta)',
        'ip2':'jet_ip2',
        'ip2_c':'jet_ip2_c',
        'ip2_cu':'jet_ip2_cu',
        'ip3':'jet_ip3',
        'ip3_c':'jet_ip3_c',
        'ip3_cu':'jet_ip3_cu',
        'sv1_ntkv':'jet_sv1_ntrkv',
        'sv1_mass':'jet_sv1_m',
        'sv1_efrc':'jet_sv1_efc',
        'sv1_n2t':'jet_sv1_n2t',
        'sv1_Lxy':'jet_sv1_Lxy',
        'sv1_L3d':'jet_sv1_L3d',
        'sv1_sig3':'jet_sv1_sig3d',
        'sv1_dR': 'jet_sv1_dR',
        'jf_n2tv':'jet_jf_n2t',
        'jf_ntrkv':'jet_jf_ntrkAtVx',
        'jf_nvtx':'jet_jf_nvtx',
        'jf_nvtx1t':'jet_jf_nvtx1t',
        'jf_mass':'jet_jf_m',
        'jf_efrc':'jet_jf_efc',
        'jf_dR':'jet_jf_dR',
        'jf_sig3':'jet_jf_sig3d' 
    }

    print 'Initializing TMVA...'
    # -- TMVA: Initialize reader, add empty variables and weights from training
    reader = TMVA.Reader()
    for n in range(n_vars):
        reader.AddVariable(var_list[n], array('f', [0] ) )
    reader.BookMVA('BDTG akt2', weights) 

    print 'Creating feature matrix...'
    # -- Get features for each event and store them in X_test
    X_buf = []
    for event in df[[names_mapping[var] for var in var_list]].values:
        X_buf.extend(np.array([normalize_type(jet) for jet in event]).T.tolist())
    X_test = np.array(X_buf)

    print 'Evaluating!'
    # -- TMVA: Evaluate!
    twoclass_output = evaluate_reader(reader, 'BDTG akt2', X_test)

    # -- Reshape the MV2 output into event-jet format
    reorganized = match_shape(twoclass_output, df['jet_pt'])

    import cPickle
    print 'Saving new MV2 weights in {}'.format(picklename)
    cPickle.dump(reorganized, open(picklename, 'wb')) 
    
    # -- Write the new branch to the tree (currently de-activated)
    #add_branch(reorganized, filename, treename, 'jet_mv2c20_new')

    print 'Done. Success!'
    return 0

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

def normalize_type(t):
    '''
    bring everything to list -- needed because some branches are arrays and others are lists
    Args:
    -----
        t: an object that could be a list or an array
    Returns:
    --------
        a list version of the input t
    '''
    if isinstance(t, list):
        return t
    else:
        return t.tolist()

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

def add_branch(arr, filename, tree = 'bTag_AntiKt2PV0TrackJets', branchname = 'jet_mv2c20_new'):
    '''
    writes the newly evaluated mv2 scores into a branch in a Friend Tree
    # --------------------------------------------------------------------------------
    # -- *WARNING*: Make sure the file you are trying to modify is *NOT* already open!
    #               Otherwise, instead of adding a branch to that file, you will
    #               corrupt the file!
    # --------------------------------------------------------------------------------
    Args:
    -----
        arr:        array containg newly evaluated mv2 scores
        filename:   .root file where new branch will be added
        tree:       (optional) name of TTree that will get a new Friend
        branchname: (optional) name of the new branch 
    '''
    # -- Check if file already exists:
    if not os.path.exists(filename):
        print '[WARNING] file not found, creating new file'
        
    # -- Open file:    
    f = root_open(filename, "update")
    # -- Extract TTree
    T = f[tree]
    
    # -- Need to figure out dtype in order to save the branch correctly
    # -- If dtype is wrong, ROOT returns garbage, so this below is important!
    
    # -- Case of branch being event level values:
    if 'float' in str(type(arr[0])):
        if '64' in str(type(arr[0])):
            dtype = 'double'
        else:
            dtype = 'float'        
    elif 'int' in str(type(arr[0])):
        dtype = 'int'
    
    # -- Case of branch being jet level list:
    elif hasattr(arr[0], '__iter__'):
        if 'float' in str(type(arr[0][0])):
            if '64' in str(type(arr[0][0])):
                dtype = 'double'
            else:
                dtype = 'float'
        elif 'int' in str(type(arr[0][0])):
            dtype = 'int'
        else:
            raise TypeError('Nested type `{}` not supported'.format(str(type(arr[0][0]))))
        dtype = 'vector<{}>'.format(dtype)
        
        
    else:
        raise TypeError('Type `{}` not supported'.format(str(type(arr[0]))))
    sys.stdout.write('Detected dtype: {}\n'.format(dtype))
    sys.stdout.flush()
    
    # -- Create friend:
    T_friend = Tree(tree+'_Friend')
    
    # -- Add new branch to friend tree:
    T_friend.create_branches({branchname : dtype})
    
    # -- Fill the branch:
    sys.stdout.write('Filling branch "{}" ... \n'.format(branchname))
    sys.stdout.flush()
    
    for i, branch4event in enumerate(arr):
        if 'vector' in dtype:
            buf = stl.vector(dtype.replace('vector<', '').replace('>', ''))()
            _ = [buf.push_back(e) for e in branch4event]
            exec('T_friend.{} = buf'.format(branchname))
            
        else:
            exec('T_friend.{} = branch4event'.format(branchname))
        T_friend.Fill()
    
    
    # -- Write out the tree and close the file:  
    sys.stdout.write('Finalizing and closing file "{}" \n'.format(filename))
    sys.stdout.flush()
    
    T.AddFriend(T_friend, tree+'_Friend')
    T_friend.Write()
    f.Write()
    f.Close()

# ----------------------------------------------------------------- 

if __name__ == '__main__':

    # -- read in arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="the .xml file from the MV2 training")
    parser.add_argument("picklename", help="path to the .pkl file with the evaluation results")
    parser.add_argument("filename", help="input .root file name")
    parser.add_argument("--tree", help="jet collection (tree) name")
    args = parser.parse_args()
    
    print "Filename = {}".format(args.filename)

    # -- pass arguments to main
    if (args.tree is not None):
        sys.exit( main(args.weights, args.picklename, args.filename, args.tree) )
    
    else:
        sys.exit( main(args.weights, args.picklename, args.filename) )

        

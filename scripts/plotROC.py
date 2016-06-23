import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandautils as pup
from sklearn.metrics import roc_curve
import cPickle

def plotROC(test_ntuple_path, picklename):
	'''
	Definition:
	-----------
		Plot a ROC curve comparison between the old mv2c10 contained in the branch and the newly evaluated one,
		which is loaded in from a pickle file.
		Both the .root file and the pickled mv2 array are assumed to be event-flat, not jet-flat.

	Args:
	-----
		test_ntuple_path: string, the path to the root files used for evaluation
		picklename: string, the path to the pickle containing the new output of your retrained mv2
	'''

	# -- import the root file into a df
	df = pup.root2panda(test_ntuple_path, 'bTag_AntiKt2PV0TrackJets')
	# -- extract the old mv2c10 branch for comparison
	oldMV2 = pup.flatten(df['jet_mv2c10'])
	# -- extract the truth labels
	truthflav = pup.flatten(df['jet_LabDr_HadF'])

	# -- open the pickle produced by evaluate_and_store
	newMV2 = pup.flatten(cPickle.load(open(picklename, 'rb')))

	# -- this allows you to check performance on b VS light
	# -- change it, if you want to look at a different performance
	bl_selection = (truthflav == 0) | (truthflav == 5)

	# -- calculate the points that make up a roc curve
	old_fpr, old_eff, _ = roc_curve(truthflav[bl_selection], oldMV2[bl_selection], pos_label=5)
	new_fpr, new_eff, _ = roc_curve(truthflav[bl_selection], newMV2[bl_selection], pos_label=5)

	# -- PLOTTING!
	# -- settings
	matplotlib.rcParams.update({'font.size': 18})
	fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

	# -- add as many curves as you want here
	# -- note: to plot rejection, take 1/false_positive_rate
	plt.plot(old_eff, 1/old_fpr, label='mv2c10 branch') 
	plt.plot(new_eff, 1/new_fpr, label='mv2c10 new')

	# -- more settings
	plt.xlim(xmin=0.6)
	plt.yscale('log')
	plt.ylim(ymax=3000)
	plt.grid(which='both')
	plt.legend() # display legend on plot
	plt.show() # open window to show plot
	fig.savefig('ROC.pdf') # save plot as a pdf

# -----------------------------------------------------------------    

if __name__ == '__main__':

	import sys
	import argparse

    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="input .root file name")
    parser.add_argument("picklename", help="path to the .pkl file with the evaluation results")
    args = parser.parse_args()

    # -- pass arguments to main
    sys.exit(plotROC(args.filename, args.picklename))






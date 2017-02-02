import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandautils as pup
from sklearn.metrics import roc_curve
import cPickle

def plotROC(test_ntuple_path):#, picklename):
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
	print 'Opening files'
	df = pup.root2panda(test_ntuple_path, 'bTag_AntiKt2PV0TrackJets', branches=['jet_mv2c10', 'jet_LabDr_HadF'])
	# -- extract the old mv2c10 branch for comparison
	oldMV2 = pup.flatten(df['jet_mv2c10'])
	# -- extract the truth labels
	truthflav = pup.flatten(df['jet_LabDr_HadF'])

	# -- open the pickle produced by evaluate_and_store
	print 'Importing pickle'
	c00 = pup.flatten(cPickle.load(open('val_Alessandro_c00.pkl', 'rb')))
	c07 = pup.flatten(cPickle.load(open('val_Alessandro_c07.pkl', 'rb')))
	c15 = pup.flatten(cPickle.load(open('val_Alessandro_c15.pkl', 'rb')))



	# -- this allows you to check performance on b VS light
	# -- change it, if you want to look at a different performance
	print 'Slicing'
	bl_selection = (truthflav == 0) | (truthflav == 5)
	print 'Plotting'
	plot(bl_selection, 'bl', truthflav, oldMV2, c00, c07, c15)

	print 'Slicing'
	bc_selection = (truthflav == 4) | (truthflav == 5)
	print 'Plotting'
	plot(bc_selection, 'bc', truthflav, oldMV2, c00, c07, c15)
	

def plot(selection, ID, truthflav, oldMV2, c00, c07, c15):

	# -- calculate the points that make up a roc curve
	old_fpr, old_eff, _ = roc_curve(truthflav[selection], oldMV2[selection], pos_label=5)
	c00_fpr, c00_eff, _ = roc_curve(truthflav[selection], c00[selection], pos_label=5)
	c07_fpr, c07_eff, _ = roc_curve(truthflav[selection], c07[selection], pos_label=5)
	c15_fpr, c15_eff, _ = roc_curve(truthflav[selection], c15[selection], pos_label=5)

	# -- PLOTTING!
	# -- settings
	matplotlib.rcParams.update({'font.size': 18})
	fig = plt.figure(figsize=(11.69, 8.27), dpi=100)

	# -- add as many curves as you want here
	# -- note: to plot rejection, take 1/false_positive_rate
	plt.plot(old_eff, 1/old_fpr, label='mv2c10 branch', color='black') 
	plt.plot(c00_eff, 1/c00_fpr, label='new c00')
	plt.plot(c07_eff, 1/c07_fpr, label='new c07')
	plt.plot(c15_eff, 1/c15_fpr, label='new c15')



	# -- more settings
	plt.xlim(xmin=0.6)
	plt.yscale('log')
	plt.xlabel(r'$\varepsilon_b$')

	if ID == 'bl':
		plt.ylabel((r'$1/\varepsilon_u$'))
		plt.ylim(ymax=1000)
	elif ID == 'bc':
		plt.ylabel((r'$1/\varepsilon_c$'))
		plt.ylim(ymax=20)

	plt.grid(which='both')
	plt.legend() # display legend on plot
	plt.show() # open window to show plot
	fig.savefig('ROC'+ID+'.pdf') # save plot as a pdf

# -----------------------------------------------------------------    

if __name__ == '__main__':

	import sys
	import argparse
	
	# -- read in arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", help="input .root file name")
	#parser.add_argument("picklename", help="path to the .pkl file with the evaluation results")
	args = parser.parse_args()
	
	# -- pass arguments to main
	sys.exit(plotROC(args.filename))#, args.picklename))
	





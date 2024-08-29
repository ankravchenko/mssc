import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import sys

cg_type='fft'
nat=sys.argv[1] #natural or artificial
#nat='natural'

label_fontsize=18

dset_natural=['interior_design', 'objects', 'scenes']
dset_artificial=['advertisement', 'art', 'infographics', 'suprematism']
colours_natural=['orange', '#4169e1', '#20b2aa']
colours_artificial=[	'#00bfff', "#701fb8", '#2f4f4f', '#ec5b0d']

#pickle load
features = [f"s{i}" for i in range(10)]


if nat == 'natural':
	dset=dset_natural
	colours=colours_natural
else:
	dset=dset_artificial
	colours=colours_artificial

plt.clf()
plt.vlines(1, 0, 0.1, colors='gray', linestyles='dashed')
plt.vlines(7, 0, 0.1, colors='gray', linestyles='dashed')
#pickle load

cnt=0
for subset_name in dset:
	#handle=open('calculated_mssc_new/'+cg_type+'_'+subset_name+'_complexity.pickle','rb')
	df=pd.read_csv('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')#pickle.load(handle)
	#average the df
	y_all=df[features].to_numpy()
	y=y_all.mean(0)
	x=range(y.size)
	plt.ylim(0,0.06)
	plt.plot(x, y, color=colours[cnt], label=subset_name)
	plt.xticks(ticks=[0,2,4,6,8],labels=[256, '75', '22', '6.5', '1.8' ]) #[256,128,96,64,48,32,24,16,12,8,4,2]. geomspace: array([256.        , 138.24764658,  74.65785853,  40.3174736 ,21.77264   ,  11.75787594,   6.34960421,   3.42897593, 1.85174942,   1.        ]) rotation=45, 
	plt.xlabel('low pass filter radius',fontsize=label_fontsize)
	plt.ylabel('partial complexity',fontsize=label_fontsize)
	plt.legend(loc='upper right', prop = { "size": label_fontsize-1 })
	cnt=cnt+1


#plt.title('correlation to human ranking by spatial scale', fontsize=label_fontsize+2)
plt.savefig('results/'+nat+'_scale_complexity_10.png', bbox_inches='tight')
	


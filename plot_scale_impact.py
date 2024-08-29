import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import sys

cg_type='fft'
subset_name='art'
nat=sys.argv[1] #natural or artificial
#nat='natural'

label_fontsize=18

dset_natural=['interior_design', 'objects', 'scenes']
dset_artificial=['advertisement', 'art', 'infographics', 'suprematism']
colours_natural=['orange', '#4169e1', '#20b2aa']
colours_artificial=[	'#00bfff', "#701fb8", '#2f4f4f', '#ec5b0d']



if nat == 'natural':
	dset=dset_natural
	colours=colours_natural
else:
	dset=dset_artificial
	colours=colours_artificial

plt.clf()
#plt.vlines(2, 0, 1, colors='gray', linestyles='dashed')
#plt.vlines(38, 0, 1, colors='gray', linestyles='dashed')
#pickle load

cnt=0
for subset_name in dset:
	df = pd.read_csv('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	plt.ylim(0,1)
	plt.plot(x, y, color=colours[cnt], label=subset_name)
	plt.xlabel('low pass filter radius',fontsize=label_fontsize)
	plt.ylabel('Pearson\'s $\it{r}$',fontsize=label_fontsize)
	labels_num=np.geomspace(256,1,8)
	labels_str=[f'{a:.1f}' for a in labels_num]
	plt.xticks(ticks=np.arange(0,40,5),labels=labels_str)
	plt.legend(loc='upper right', prop = { "size": label_fontsize-1 })
	cnt=cnt+1

#plt.title('correlation to human ranking by spatial scale', fontsize=label_fontsize+2)
plt.savefig(nat+'_scale_impact_100.png')


#repeat for broader strides


plt.clf()
plt.vlines(1, 0, 1, colors='gray', linestyles='dashed')
plt.vlines(7, 0, 1, colors='gray', linestyles='dashed')
#pickle load

cnt=0
for subset_name in dset:
	df = pd.read_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	plt.ylim(0,1)
	plt.plot(x, y, color=colours[cnt], label=subset_name)
	plt.xlabel('low pass filter radius',fontsize=label_fontsize)
	plt.ylabel('Pearson\'s $\it{r}$',fontsize=label_fontsize)
	plt.xticks(ticks=[0,2,4,6,8],labels=[256, '75', '22', '6.5', '1.8' ])
	plt.legend(loc='upper right', prop = { "size": label_fontsize-1})
	cnt=cnt+1

#plt.title('correlation to human ranking by spatial scale', fontsize=label_fontsize+2)
plt.savefig(nat+'_scale_impact_10.png', bbox_inches='tight')
	


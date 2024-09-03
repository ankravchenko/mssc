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
plt.vlines(1, 0, 1, colors='gray', linestyles='dashed')
plt.vlines(8, 0, 1, colors='gray', linestyles='dashed')
#pickle load


blur_r=np.geomspace(256.0, 1.0,num=11)
ticks_r=[0,2,4,6,8,10]
labels_r=[]
for r in ticks_r:
    a=blur_r[r]  
    labels_r.append(f'{a:.1f}')

cnt=0
for subset_name in dset:
	df = pd.read_csv('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	plt.ylim(0,1)
	plt.plot(x[0:10], y[0:10], color=colours[cnt], label=subset_name)
	plt.xlabel('low pass filter radius',fontsize=label_fontsize)
	plt.ylabel('Pearson\'s $\it{r}$',fontsize=label_fontsize)
	'''
	labels_num=np.geomspace(256,1,8)
	labels_str=[f'{a:.1f}' for a in labels_num]
	plt.xticks(ticks=np.arange(0,40,5),labels=labels_str)
	'''
	plt.xticks(ticks=ticks_r, labels=labels_r)
	plt.legend(loc='upper right', prop = { "size": label_fontsize-1 })
	cnt=cnt+1

#plt.title('correlation to human ranking by spatial scale', fontsize=label_fontsize+2)
plt.savefig('results/'+nat+'_scale_impact.png')


#repeat for broader strides


plt.clf()
plt.vlines(1, 0, 1, colors='gray', linestyles='dashed')
plt.vlines(8, 0, 1, colors='gray', linestyles='dashed')
#pickle load

cnt=0
for subset_name in dset:
	df = pd.read_csv('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', delimiter='\t')
	y=df['r'].to_numpy()
	x=range(y.size)
	plt.ylim(0,1)
	plt.plot(x[0:10], y[0:10], color=colours[cnt], label=subset_name)
	plt.xlabel('low pass filter radius',fontsize=label_fontsize)
	plt.ylabel('Pearson\'s $\it{r}$',fontsize=label_fontsize)
	plt.xticks(ticks=ticks_r, labels=labels_r)
	plt.legend(loc='upper right', prop = { "size": label_fontsize-1})
	cnt=cnt+1

#plt.title('correlation to human ranking by spatial scale', fontsize=label_fontsize+2)
plt.savefig('results/'+nat+'_scale_impact.png', bbox_inches='tight')
	


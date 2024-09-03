import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import sys
import os

#read parameters
subset_name='suprematism'
subset_name=sys.argv[1]
cg_type='fft'
#cg_type=sys.argv[2]

#pickle load
handle=open('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.pickle','rb')
df=pickle.load(handle)

#remove outliers

if subset_name=='suprematism':
	idx=[50]#artifacts from poor scanning quality comparable to meaningful features
	df=df.drop(idx)

if subset_name=='advertisement': 
	idx=[199]#mcdonalds logo was ranked as 0 by humans while being technically somewhat complex
	df=df.drop(idx)

'''if subset_name=='infographics':
	idx=[115, 99, 25, 117, 44, 68, 177, 140, 35]#these only have text. removing them improves performance but doesn't alter distribution shape
	outliers=df.loc[idx]
	df=df.drop(idx)'''

'''
idx=[]
#[13,20,50,53,64]
if subset_name=='suprematism':
	idx=[13, 20, 48, 50, 52, 53, 64]#[50, 95, 64, 65, 52, 53, 13, 48, 20]
elif subset_name=='advertisement':
	#msk=df['ms_total']>0.3 
	idx = [85, 86]#df.index[msk] 109
elif subset_name=='art':
	idx=[30, 88, 107, 119, 225, 286, 380, 200]
	#idx=[119,107,88,30,200, 225,380, 328, 286, 216, 316, 332, 86, 108, 73, 23, 285,31]
	#idx=[328, 80, 86, 225, 286, 352, 380, 119] for tasteless images
	#idx=[380, 225, 30, 88, 119, 286, 107] for broad lines
#elif subset_name=='scenes':
	#idx=[52, 22]
#elif subset_name=='infographics':
	#idx=[43,48,63,103,105,171,173]
	#msk=df['ms_total']>0.5 
	#idx = df.index[msk]
'''


'''
x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

y1o=outliers['gt'].to_numpy()
y2o=outliers['ms_total'].to_numpy()
'''

#outliers=df.loc[idx]
#df=df.drop(idx)

#separate df for statistics
#regression for each separate scale

#regression for each separate scale: loop
reg_results=[]
y=df['gt']
features = [f"s{i}" for i in range(19)]

max_r=0
for i in range(14):
	#regression
	x=df[features[i]]
	#xo=outliers['ms_total']
	#yo=outliers['gt']
	if x.any():
		slope, intercept, r, p, std_err = stats.linregress(x, y)
		reg_results.append([r,p])
		if (r>max_r):
			max_r=r
	else:
		reg_results.append([0,0])
print("max r: ", str(max_r))

df_stats = pd.DataFrame(reg_results, columns=['r', 'p'])


if not os.path.exists('results/calculated_mssc/'):
	os.mkdir('results/calculated_mssc/')
if not os.path.exists('results/calculated_mssc_eps/'):
	os.mkdir('results/calculated_mssc_eps/')

df_stats.to_csv('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_regression.csv', sep='\t')
print(subset_name+', full: r=', str(r)+', p='+str(p))


fr1=2
fr2=7
ttt=df['s'+str(fr2)]
for i in range(fr1,fr2):
	ttt=ttt+df['s'+str(i)]
df['frac']=ttt
x=df['frac']
slope, intercept, r, p, std_err = stats.linregress(x, y)
freqrange=str(fr1)+'-'+str(fr2)
print(subset_name+', '+freqrange+': r=', str(r)+', p='+str(p))



#regression
x=df['frac']
y=df['gt']
#xo=outliers['frac']
#yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.ylabel('subjective complexity',fontsize=16)
plt.xlabel('MSSC',fontsize=16)
#plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.xlim([-0.01, 0.175])
plt.title(subset_name, fontsize=16)
plt.savefig('results/mssc_figures/'+cg_type+'_'+subset_name+'_regression_'+freqrange+'.png')
plt.savefig('results/mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_'+freqrange+'.eps', format='eps')

df_part=df[['gt','frac']]
df_part.to_csv('results/calculated_mssc/'+cg_type+'_'+subset_name+'_complexity_mid.csv', sep='\t')

f = open("results/mssc_figures/"+cg_type+'_'+subset_name+'_regression_'+freqrange+'.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()


#regression full
x=df['ms_total']
y=df['gt']
#xo=outliers['ms_total']
#yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.ylabel('subjective complexity',fontsize=16)
plt.xlabel('MSSC',fontsize=18)
#plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.xlim([0, 0.35])
plt.title(subset_name, fontsize=16)
plt.savefig('results/mssc_figures/'+cg_type+'_'+subset_name+'_regression.png')
plt.savefig('results/mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression.eps', format='eps')



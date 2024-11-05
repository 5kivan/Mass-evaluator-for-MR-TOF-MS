# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint
import pickle, os
import mmap
import os
import sys
import numpy as np
import pandas as pd

ame16= pd.read_csv('elements.txt', header=None, sep="	")
ame16.columns=['el','atm_int','atm_frac','err']
m_e=548.579909065*1e-6
F1=float(input('F1> '))
G1=float(input('G1> '))
M1=F1/3

alpha=float(input('alpha> '))
beta=float(input('beta> '))
N=float(input('Number of revolutions> '))

ame16['sindly_charged']=(ame16['atm_int']+ame16['atm_frac']*1e-6)-m_e
ame16['doubly_charged']=((ame16['atm_int']+ame16['atm_frac']*1e-6)-2*m_e)/2

TOF_singly=pd.DataFrame(alpha*np.sqrt(ame16['sindly_charged'])+beta)
TOF_doubly=pd.DataFrame(alpha*np.sqrt(ame16['doubly_charged'])+beta)
TOF_singly['ISEP']=TOF_singly['sindly_charged']-F1-M1
TOF_doubly['ISEP']=TOF_doubly['doubly_charged']-F1-M1
TOF_singly['TOF for N revol']=TOF_singly['ISEP']/1000*N
TOF_doubly['TOF for N revol']=TOF_doubly['ISEP']/1000*N
TOF_singly['TOF total']=(TOF_singly['TOF for N revol']+F1+M1)*1000
TOF_doubly['TOF total']=(TOF_doubly['TOF for N revol']+F1+M1)*1000
TOF_singly['El']=ame16['el']+str(1)
TOF_doubly['El']=ame16['el']+str(2)
'''
lst_files = ['data/test_run455.lst',    # nrevs = x, bg = x ms, cycles = x, test mode
             'data/test_run456.lst']    # nrevs = x, bg = x ms, cycles = x, test mode

# open preanalyzed dataset if existing, raw conversion takes LONG!
if os.path.isfile('mr-tof-data.p'):
    df = pickle.load(open('mr-tof-data.p','rb'))
else:
    df = process_lst('test_run455.lst')
    pickle.dump(df, open('mr-tof-data.p','wb'))

'''
data_whole = pd.read_csv('Sc_Run459.mpa', header=172, float_precision='high')
data_whole[['tof',"sweep", "counts"]] = pd.DataFrame([ x.split() for x in data_whole["[DATA]"].tolist() ])
data_whole = data_whole.drop('[DATA]', axis=1).astype(float)
tof_window = []

#computing the tof value
caloff=22190995.200000
calfact=0.8
data_whole['tof']=(caloff+data_whole['tof']*calfact)

if tof_window != []:
    data_whole = data_whole[(data_whole.tof >= tof_window[0]) & (data_whole.tof <= tof_window[1])]

fig, ax = plt.subplots(constrained_layout=True)
#ax_0.legend('data',loc='upper right')
ax.set_xlabel('')
ax.ticklabel_format(useOffset=False, style='plain')
x_proj = data_whole.tof.value_counts(bins=20032).sort_index()
ax.bar(x_proj.index.mid, x_proj, width = 10,color='black')
ax.set_yscale('log')
ax.set_ylabel('counts / 8 ns', fontsize=24, fontweight='bold')


for i in range(0,len(TOF_singly['TOF total'])):
    if min(data_whole.tof)<=TOF_singly['TOF total'][i]<=max(data_whole.tof):
        print(min(data_whole.tof),max(data_whole.tof))
        plt.axvline(TOF_singly['TOF total'][i],c='#%06X' % randint(0, 0xFFFFFF))
        print(TOF_singly['El'][i])
        print(TOF_singly['TOF total'][i])
        #plt.text(TOF_singly['TOF total'][i], -2, TOF_singly['El'][i], ha='right', va='top',c='red' )
plt.savefig("evaluator.pdf",bbox_inches="tight")     

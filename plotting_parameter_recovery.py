# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:31:53 2022

@author: u0141056
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_plot = pd.read_csv('parameter_recoveryAR99.csv')


fig, ax = plt.subplots(1,5)

plt.subplots_adjust(left=-.5,
                    bottom=0.1, 
                    right=1, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

def scatter(x,y,axis):
    p = sns.scatterplot(data=df_plot, x=x, y=y,hue = "ntrials",ax=ax[axis])
    
    # fund the min and max value to determine the range of x and y axis
    low = min(df_plot[x] + df_plot[y])
    high = max(df_plot[x] + df_plot[y])
    
    # add some extra space to min and max (such that extreme points are still clearly visible)
    llow = low - (abs(high - low)/2)
    hhigh = high + (abs(high - low)/2)
    p.set_ylim(llow, hhigh)
    p.set_xlim(llow, hhigh)
    
    p.set_aspect('equal', adjustable='box')

    # Draw a line of x=y 
    lims = [llow, hhigh]
    p.plot(lims, lims, '-r')

scatter("sens_sim","sens_fit",0)
scatter("bias_sim","bias_fit",1)
scatter("sigma_sim","sigma_fit",2)
scatter("pc_sim","pc_fit",3)
scatter("pe_sim","pe_fit",4)

     
fig.show()



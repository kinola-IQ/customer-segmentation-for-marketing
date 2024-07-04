# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:28:21 2024

@author: akinola
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly 
import seaborn

#---------module to plot the viduals------

def box(df,cols,by,grid=True):
    """
    This checks for outliers by creating box plots.
    Parameters: 
    df : DataFrame
        The DataFrame containing the data.
    cols : str or list of str
        The column(s) to plot.   
    by : str
        The column name to group by.
    grid : bool
        Whether to show the grid or not.
    """
    df.boxplot(column=cols,by=by)
    
def hist(df,index):
    # df[[index]].plot.hist()
    df[index].plot.hist()
    plt.figure()
    plt.show()
    
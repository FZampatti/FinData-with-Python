'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
The code below has been written for the Financial Data Science with Python
course at Birkbeck, University of London, in Spring 2021.

Author: Filippo Zampatti

The project coding includes four files:
    Get_data
    NSS
    PCA
    AE

All files should be open at the same time for best performance.
Once Get_data is run, all other files should work as standalone analyses.

NOTE: many error warnings will appear on the left side of the Spyder console
      The Spyder developing team is working on a way to allow for deactivation 
      of such errors.

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Import utilities to be used across .py files

import pandas as pd

import numpy as np
from numpy.linalg import lstsq

import scipy as sp
from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib import cm
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, date2num

##############################################################################
#######                         IMPORTANT                               ######
##############################################################################

# the url below dowloads the data from my GitHub repository
# To make it easeir, the repository is currently public and can be accessed
# to download the Excel file containg the data as well as these .py files

url = 'https://github.com/FZampatti/FinData-with-Python/'\
    'blob/main/Swap%20Data.xlsx?raw=true'

# Alternatively, download the file and use a path similar to the below:
# url = r"C:\Users\filippo\.spyder-py3\Swap Data_final.xlsx"


##############################################################################


# Import first sheet with daily data as a Pandas dataframe, then save a
# version subtracting the mean and one normalize

swap_d = pd.read_excel(url,sheet_name="Daily").set_index(['Dates'])
swap_d_demeaned = swap_d - swap_d.mean(axis=0)
swap_d_norm = (swap_d - swap_d.mean(axis=0))/swap_d.std(axis=0)

# Do the same on the second sheet (weekly data)

swap_w = pd.read_excel(url,sheet_name="Weekly").set_index(['Dates'])
swap_w_demeaned = swap_w - swap_w.mean(axis=0)
swap_w_norm = (swap_w - swap_w.mean(axis=0))/swap_w.std(axis=0)

# Do the same on the third sheet (monthly data)

swap_m = pd.read_excel(url,sheet_name="Monthly").set_index(['Dates'])
swap_m_demeaned = swap_m - swap_m.mean(axis=0)
swap_m_norm = (swap_m - swap_m.mean(axis=0))/swap_m.std(axis=0)

# Extract last observation (16th April 2021) for each transformation

last_data = swap_d.iloc[-1].to_numpy()
last_data_demeaned = swap_d_demeaned.iloc[-1,:].to_numpy()
last_data_norm = swap_d_norm.iloc[-1,:].to_numpy()


##############################################################################
'''
The code below allows to plot the historical yield curve as a 3D surface
'''
# Convert dates format to plottable numbers
surf_date = date2num(swap_d.index)

# First, compute a grid of the correct size
X = surf_date
Y = swap_d.columns.str.replace('[USSW]', '').astype('int64')
X, Y = np.meshgrid(Y, X) # creates 2 dimensional grid

# Data to be plotted (the yield curve)
Z = np.array(swap_d, dtype='f')

# Set up 3D figure

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20,11))

# Plot the surface
surf = ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Convert string back to date
loc = AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()

# Make background transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Increase font size for tick labels
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.tick_params(axis="z", labelsize=18)

# Set axis labels
ax.set_xlabel('Year',  fontsize=22, labelpad= 50)
ax.set_ylabel('Tenor', fontsize=22, labelpad=20)
ax.set_zlabel('Yield (%)',  fontsize=22, labelpad=10)

# Show the plot
plt.show()

##############################################################################

# END
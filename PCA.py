'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
The code below has been written for the Financial Data Science with Python
course at Birkbeck, University of London, in Spring 2021.

Author: Filippo Zampatti

The project coding includes four files:
    Get_data
    NS
    PCA
    AE

All files should be open at the same time for best performance.
Once Get_data is run, all other files should work as standalone analyses.


This file runs the PCA decomposition to the USD swap data both by manually 
computing the eigenvalues and eigenvectors, as well as using the ScikitLearn 
Package PCA()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Define function to compute manually the eigenvalues and eigenvectors.
# It also returns the historical levels of the factors
# Note that the data input should be the RAW data, as the demeaning is carried
# out within the function 

def myPCA (data, dim):
    data_demeaned = data - data.mean(axis=0)
    Cov = np.cov(data_demeaned, rowvar=False)
    eigenvals, eigenvecs = sp.linalg.eigh(Cov)
    eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]]
    eigenvals = eigenvals[np.argsort(eigenvals)[::-1]]
    eigenvecs = eigenvecs[:, :dim]
    return np.dot(eigenvecs.T, data_demeaned.T).T, eigenvals, eigenvecs

dim = 3 # Set the number of factors to be retained as 3

scores, evals, evecs = myPCA(swap_w, dim) # Save PCA decomposition results

# Compute the PCA-reconstructed yield curve data series
reconst = pd.DataFrame(np.dot(scores,evecs.T), index=swap_w.index,
                       columns=swap_w.columns)

# Add back the mean to obtain the actual yield levels

for cols in reconst.columns:
    reconst[cols] = reconst[cols] + swap_w.mean(axis=0)[cols]

### Now run the PCA decomposition using the ScikitLearn Package

pca = PCA()
pca.fit(swap_w) # run the decomposition and store it

# Plot the variance explained by each component

fig, ax = plt.subplots()

fig.set_size_inches(5,2)
ax.plot(np.arange(1,16), pca.explained_variance_ratio_*100)
ax.set_title("Variance Explained")
ax.set_xlabel("Eigenvalues")
ax.set_xticks(np.arange(1,16))
ax.set_xticklabels(np.arange(1,16))
ax.set_ylabel("%")


# Plot the first three components against tenors.
# It shows that the components correspond to level, slope and curvature.

fig, ax = plt.subplots()

ax.plot(pca.components_[0:3].T)
ax.set_title("First Three Components")
ax.set_xlabel("Tenor")
ax.set_xticks(np.arange(0,15))
ax.set_xticklabels(np.arange(1,16))
ax.set_ylabel("Level")


'''
The code below was sed to assess how much variance is explained by each factor
over time, using a 3 month rolling window of weekly observations
'''

window = 13 # in weeks, 3 months
HistPCA = np.empty([swap_w.shape[0]-window,3]) # define matrix

istart = swap_w.shape[0]

# Set up loop to compute PCA recursively
for i in range(window,swap_w.shape[0]):
    
    iend = istart-window
    pcahist = pca.fit(swap_w.iloc[iend:istart, :])
    HistPCA[i-window] = pcahist.explained_variance_ratio_[0:3]
    istart = istart-1
    
# This aspect is not explored further, but a quick check on the resulting 
# matrix shows how the first three components explain more than 99% of the 
# yield curve moves in the past 20 years.

''' 

PLOTTING SECTION

Here below, the code for the charts used in the project pdf is shown

'''
### Plot outright interest rate vs first component

PCA_5y = pd.DataFrame(swap_w["USSW5"]).rename(columns={"USSW5":"5y Swap"})
PCA_5y['First Factor'] = -scores[:,0]
PCA_5y.index.names = ['Year']
left_5y = PCA_5y.plot(y=["5y Swap"])
left_5y.set_ylabel("Swap Rate (%)")
ax = PCA_5y.plot(y=["First Factor"], secondary_y=True, ax=left_5y,
               title='5y Outright vs First Factor')
ax.set_ylabel("Level")

PCA_5y.corr() # Can check that the correlation between the series is high

### Plot curve strategy vs second component

PCA_1s30s = pd.DataFrame(swap_w["USSW30"]-swap_w["USSW1"])
PCA_1s30s.columns = ["1s30s Curve"]
PCA_1s30s.index.names = ['Year']
PCA_1s30s['Second Factor'] = scores[:,1]
left_1s30s = PCA_1s30s.plot(y=["1s30s Curve"])
left_1s30s.set_ylabel("Swap Rate (%)")
ax = PCA_1s30s.plot(y=["Second Factor"], secondary_y=True, ax=left_1s30s,
                  title='1s30s Curve vs Second Factor')
ax.set_ylabel("Level")

PCA_1s30s.corr() # Can check that the correlation between the series is high

### Plot fly strategy vs third component

PCA_1s5s30s = pd.DataFrame(swap_w["USSW5"]*2-swap_w["USSW1"]-swap_w["USSW30"])
PCA_1s5s30s.columns = ["1s5s30s Fly"]
PCA_1s5s30s['Third Factor']= -scores[:,2]
left_1s5s30s = PCA_1s5s30s.plot(y=["1s5s30s Fly"])
left_1s5s30s.set_ylabel("Swap Rate (%)")
ax = PCA_1s5s30s.plot(y=["Third Factor"], secondary_y=True, ax=left_1s5s30s,
                    title='1s5s30s Fly vs Third Factor')
ax.set_ylabel("Level")

PCA_1s5s30s.corr() # Can check that the correlation between the series is high

### Plot fitted curve vs last data point available

fig, ax = plt.subplots()

ax.plot(t,last_data, 'o', markerfacecolor = 'white')
ax.plot(t,reconst.iloc[-1])
ax.set_xlabel("Tenor")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("PCA Curve vs Swaps")

### Plot the time series of the outright rate vs the PCA reconstructed rate

# First set up data frame containing the two series
PCAvsReal= pd.DataFrame(swap_w["USSW10"]).rename(columns={"USSW10":"10y Swap"})
PCAvsReal['PCA Reconstructed'] = reconst.loc[:,'USSW10']

# Plot the data frame
ax = PCAvsReal.plot(y=["10y Swap","PCA Reconstructed"], style=['-','--'],
                  title='10y Outright vs PCA reconstructed')
ax.lines[1].set_alpha(0.8) 
ax.set_ylabel("Swap Rate (%)")

##############################################################################

# END
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


This file runs the Autoencoder analysis to the USD swap data using the 
MPLRegressor module within the ScikitLearn Package.

Please refer to the project pdf for a description of the parameters used. 

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

### Define autoencoder structure by setting all relevant parameters

autoencoder = MLPRegressor(hidden_layer_sizes = 3, 
                           activation = 'tanh', 
                           solver = 'adam', 
                           learning_rate_init = 0.00005, 
                           max_iter = 50000, 
                           tol = 0.000001, 
                           verbose = True) # Set verbose to False to hide the 
                                           # iteration process printing

# Set seed for replication

np.random.seed(168) 

# fit autoencoder to daily data (please run together with the seed above to
# obtain the same values of the pdf )

# We need to use normalized data for best fit
 
fitAE = autoencoder.fit(swap_d_norm,swap_d_norm)


# Plot the three output loadings of the autoencoder

fig, ax = plt.subplots()
plot(t, fitAE.coefs_[1].T)
ax.set_xlabel("Tenor")
ax.set_ylabel("Factor level")
ax.set_title("Autoencoder Output Loadings")

# Compute R-squared of the autoencoder on the monthly dataset. It's very high.

fitAE.score(swap_m_norm, swap_m_norm)

# Fit the autoencoder to the last data observation
pred = fitAE.predict(last_data_norm.reshape(1, -1))*\
       swap_d.std(axis=0).to_numpy() + swap_d.mean(axis=0).to_numpy()

# Fit the autoencoder to the entire weekly data set. Similar to the PCA
# reconstruction done in the PCA file

pred_series_w= fitAE.predict(swap_w_norm)*swap_w.std(axis=0).to_numpy()\
               + swap_w.mean(axis=0).to_numpy()

''' 
PLOTTING SECTION

Here below, the code for the charts used in the project pdf is shown

'''
### Plot fitted curve vs last data point available

fig, ax = plt.subplots()

ax.plot(t, last_data, 'o',markerfacecolor='white')
ax.plot(t, pred.T)
ax.set_xlabel("Tenor")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("Autoencoder Curve vs Swaps")

### Plot the time series of the outright rate vs the PCA reconstructed rate

# First set up data frame containing the two series
AEvsReal = pd.DataFrame(swap_w["USSW10"]).rename(columns={"USSW10":"10y Swap"})
AEvsReal['Autoencoder Reconstructed'] = pred_series_w[:, 9]

# Plot the data frame
ax = AEvsReal.plot(y=["10y Swap", "Autoencoder Reconstructed"],
                 style=['-','--'], title='10y Outright vs AE reconstructed')
ax.lines[1].set_alpha(0.8) 
ax.set_ylabel("Swap Rate (%)")


# Construct fictitious data point by setting the long end of the curve to a 
# constant level

fake_series = np.array([0.215 , 0.2789, 0.4667, 0.6982, 0.9122, 1.0989,
                        1.2552, 1.3795, 1.4815, 1.5664, 1.5664, 1.5664,
                        1.5664, 1.5664, 1.5664])

# normalize the series
fake_series_norm = (fake_series-swap_d.mean(axis=0).to_numpy())/ \
                   swap_d.std(axis=0).to_numpy()

# Predict the yield curve shape using the three models applied to the fake
# data. 

pred_fake_AE = fitAE.predict(fake_series_norm.reshape(1, -1))* \
               swap_d.std(axis=0).to_numpy()+ swap_d.mean(axis=0).to_numpy()
pred_fake_PCA = np.dot(np.dot(evecs.T, (fake_series-swap_d.mean(axis=0)\
                .to_numpy()).T).T,evecs.T)+ swap_w.mean(axis=0)
                # from PCA file
pred_fake_NS = FitCurve(t, fake_series, t)[0] # from NSS file

# Plot the three predicted curves against the fake (distorted curve) and the 
# original observed curve

# It seems like the autoencoder outperforms in predicting the curve on
# fake data.

fig, ax = plt.subplots()

ax.plot(t, fake_series, label='Fake')
ax.plot(t, last_data, '--', color='black', label='Real' )
ax.plot(t, pred_fake_AE.T, label='AE')
ax.plot(t, pred_fake_PCA, label='PCA')
ax.plot(t, pred_fake_NS, label='NS')
ax.set_xlabel("Tenor")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("Estimation Accuracy from Distorted Curve")
leg = ax.legend()

##############################################################################

# END
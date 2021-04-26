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


This file calibrates the Nelson Siegel model to the USD swap data

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Set an initial tau to be used to run OLS on factors and obtain Beta
tau = 1

# Extract tenors (in years) as numerical values
t = swap_d.columns.str.replace('[USSW]', '').astype('int64')

# Extract tenors (in years) as a list
tenors = swap_d.columns.str.replace('[USSW]', '').tolist()

# Create a evenly spaced range to obtaine smoother factors and curves, helping
# visualization
linspace = np.arange(1, 30.1, 0.1)


# Define function to compute factors across tenors, given a tau value
def Factors(tau, T):
        exp_term = np.exp(-T/tau)
        factor0 = np.ones(T.size)
        factor1 = (1 - exp_term) / (T / tau)
        factor2 = factor1 - exp_term
        return np.stack([factor0, factor1, factor2]).transpose()

# Define function to estimate OLS betas, given a tau value and a data point
# i.e. the yield curve for a given day.
# Note that the size of "tenors" and "data" inputs need to match
   
def Betas (tau, tenors, data):
        myfactors = Factors(tau = tau, T = tenors)
        lstsq_res = lstsq(myfactors, data, rcond = None)
        beta = lstsq_res[0]
        return beta[0], beta[1], beta[2]

# Define function to compute the curve with OLS betas, given a tau value.
# Relies on "Factors" and "Betas" functions, make sure to run them!
    
def Curve (tau, tenors, data):
        myfactors = Factors(tau = tau, T = tenors)
        beta = Betas(tau = tau,tenors = tenors, data = data)
        return beta[0]+beta[1]*myfactors.transpose()[1]+beta[2]*myfactors.\
               transpose()[2]

# Define function to compute the squared error of the OLS curve and data.
# Relies on "Curve" function.

def Sqerror(tau, T, data):
        sqerror= np.sum((Curve(tau = tau,tenors = T,data = data)-data)**2)
        return sqerror
    
# Define function to calibrate the curve by finding the tau parameter which
# minimizes the squared error. It then re-estimates OLS betas with the optimal
# tau.
# Relies on "Sqerror", "Betas" and "Factors" functions.
    
def Calibrate (tenors, data, tau1 = 1):
            opt_res = minimize(Sqerror, x0=tau1, args=(tenors, data),
                               bounds=[(1.0, 10.0)])
            targetbeta = Betas(opt_res.x[0], tenors, data)
            newfactors = Factors(tau = opt_res.x[0], T = tenors)
            return targetbeta, opt_res.x[0],newfactors

# Define function to find the best fitting curve given a specific data point
# tenors input needs to match the data dimensions 
# T can be constructed with more tenors to create a smoother curve.
# For example, use the linspace defined above


def FitCurve (tenors,data, T):
    beta = Calibrate(tenors, data)[0]
    factors = Factors(Calibrate(tenors, data)[1],T)        
    fittedcurve = beta[0]+beta[1]*factors.transpose()[1]+beta[2]*factors.\
                  transpose()[2]
    F1 = beta[0]*factors.transpose()[0]
    F2 = beta[1]*factors.transpose()[1]
    F3 = beta[2]*factors.transpose()[2]
    return fittedcurve, beta, factors, F1, F2, F3
    
''' 
Run a loop across weekly yield curve observation to see how the beta parameters
evolve during time.

Store the result in a Numpy array

Requires a couple of minutes of running time

'''


HistBetas = np.empty([swap_w.shape[0],3])

for i in range(0, swap_w.shape[0]):
    coeff = FitCurve(t, swap_w.iloc[i].to_numpy(), t)[1]
    HistBetas[i] = np.asarray(coeff)

# To quickly check the evolution of betas over time, run the basic plot below

# plot(HistBetas)

''' 

PLOTTING SECTION

Here below, the code for the charts used in the project pdf is shown

'''

### Plot outright interest rate vs beta 0

outright30 = pd.DataFrame(swap_w["USSW30"]).rename(columns={"USSW30":"30y Swap"})
outright30['Beta 0'] = HistBetas[:,0]
ax = outright30.plot(y=["30y Swap","Beta 0"], title='30s Outright vs Beta 0')
ax.set_xlabel("Year")
ax.set_ylabel("Level")

outright30.corr() # Can check that the correlation between the series is high

### Plot curve strategy vs beta 1

curve_1s30s = pd.DataFrame(swap_w["USSW30"]-swap_w["USSW1"])
curve_1s30s.columns = ["1s30s Curve"]
curve_1s30s['Beta 1'] = -HistBetas[:,1]
ax = curve_1s30s.plot(y=["1s30s Curve","Beta 1"],title='1s30s Curve vs Beta 1')
ax.set_xlabel("Year")
ax.set_ylabel("Level")

curve_1s30s.corr() # Can check that the correlation between the series is high

### Plot fly strategy vs beta 2

fly_5s10s30s=pd.DataFrame(2*swap_w["USSW10"]-swap_w["USSW5"]-swap_w["USSW30"])
fly_5s10s30s.columns = ["5s10s30s Fly"]
fly_5s10s30s['Beta 2'] = -HistBetas[:,2]*0.05
ax = fly_5s10s30s.plot(y=["5s10s30s Fly","Beta 2"],\
   title='5s10s30s Fly vs Beta 2')
ax.set_xlabel("Year")
ax.set_ylabel("fly, beta*0.05")

fly_5s10s30s.corr() # Can check that the correlation between the series is high

### Plot fitted curve vs last data point available

# First, fit the curve and save it as array mapping curve to tenors
last_fitted = FitCurve(t, last_data, linspace)[0]
last_fitted = np.column_stack((linspace, last_fitted))

# Plot the two series

fig, ax = plt.subplots()

ax.plot(t, last_data, 'o', markerfacecolor = 'white')
ax.plot(last_fitted[:,0], last_fitted[:,1])
ax.set_xlabel("Tenor")
ax.set_ylabel("Swap Rate (%)")
ax.set_title("NSS Curve vs Swaps")

### Plot Zero Factors

fig, ax = plt.subplots()

ax.plot(linspace, Factors(5, linspace)) # Use tau=5 to help visualization
ax.set_xlabel("Tenor")
ax.set_ylabel("Factor Level")
ax.set_title("Factors (Zero curve)")

### Plot Fwd Factors

# First, define function to compute forward factors

def FwdFactors(tau, T):
        exp_term = np.exp(-T/tau)
        factor0 = np.ones(T.size)
        factor1 = exp_term
        factor2 = (exp_term)*(T / tau)
        return np.stack([factor0, factor1, factor2]).transpose()

# Plot the two series

fig, ax = plt.subplots()
ax.plot(linspace, FwdFactors(5, linspace)) # Use tau=5 to help visualization
ax.set_xlabel("Tenor")
ax.set_ylabel("Factor Level")
ax.set_title("Factors (Forward curve)")

##############################################################################

# END

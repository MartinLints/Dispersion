# According to H. Lamb "On waves in an elastic plate"
# He used xi for k (spatial frequency)
#         sigma for omega (radial freq.)
#         f for h/2 (half thickness)
#         Making the relevant changes we get the following code for Si and Ai

from scipy import *
from pylab import *
from joblib import Parallel, delayed



##############################################################
#############      Modifiable parameters      ################
##############################################################
rho = 7932.0 # kg/m**3
E = 216.9e9   # Pa
nu = 0.2865   # Poisson's ratio
h = 10.0e-3  # thickness of the plate 
NONDIM = True # if True, results are for f*h MHz*mm, not f.
              # effectively h=1


frmax = 15e6 # maximum frequency to be analyzed
maxmodes = 3 # no. of modes to be plotted
##############################################################
##########      End of modifiable parameters      ############
##############################################################

if NONDIM:
    h = 1.0e-3


cl = sqrt( (E*(1-nu))/(rho*(1+nu)*(1-2*nu)))
ct = sqrt( E/(2*rho*(1+nu)))

# symmetric
def Si(k, omega):
    alpha2 = k**2 - omega**2/cl**2+0j
    beta2 =  k**2 - omega**2/ct**2+0j
    return tanh(sqrt(beta2)*h/2)/sqrt(beta2) - (4*k**2*sqrt(alpha2)*tanh(sqrt(alpha2)*h/2)) / (k**2 + beta2)**2


# antisymmetric. Opposite sign, since negative-to-positive crossover is detected in
# the calcstuff function
def Ai(k, omega):
    alpha2 = k**2 - omega**2/cl**2+0j
    beta2 =  k**2 - omega**2/ct**2+0j
    return sqrt(beta2)*tanh(sqrt(beta2)*h/2) - ((k**2 + beta2)**2 * tanh(sqrt(alpha2)*h/2)) / (4*k**2*sqrt(alpha2))


kmax = 1.2*2*pi*frmax/ct # maximum k to be tested for frequencies 
# see https://www.matec-conferences.org/articles/matecconf/pdf/2018/16/matecconf_mms2018_08011.pdf
kdelta = kmax/1e6 # the finer the kdelta, the more tries. We do no refinement afterwards
ktests = arange(10, kmax, kdelta)
freqlist = linspace(1.0e4, frmax, 200) # last is num. frequencies
def calcstuff(f):
    """
    given frequency to analyze, finds the array of dispersion curve points at that freq
    returns list of two arrays. First for antisymmetric, second for symmetric
    """
    ai = []
    si = []
    omega = 2*pi*f
    cps = omega/ktests # phase speed to be tested
    residA = real(Ai(ktests, omega))
    residS = real(Si(ktests, omega))
    # from positive to negative
    ptnS = where((residS[0:-1]*residS[1:]<=0) &(residS[0:-1]>residS[1:]))[0]
    # find the cross from negative to positive
    ptnA = where((residA[0:-1]*residA[1:]<=0) &(residA[0:-1]<residA[1:]))[0]
    return [cps[ptnS], cps[ptnA]] 


freq_run = Parallel(n_jobs=-1, max_nbytes=1e8, verbose=10, backend='multiprocessing')(delayed(calcstuff)(f) for f in freqlist) # loky backend not working



#######################
# gather in dispersion curve data to lines
a_s = zeros((len(freq_run), maxmodes))*nan
s_s = zeros((len(freq_run), maxmodes))*nan
for i, f in enumerate(freq_run):
    if len(f[0]) != 0:
        speeds = sort(f[0])[:maxmodes]
        for si,sp in enumerate(speeds):
            s_s[i, si] = sp
    if len(f[1]) != 0:
        speeds = sort(f[1])[:maxmodes]
        for si,sp in enumerate(speeds):
            a_s[i, si] = sp

######################################            
# plot the lines
for i in range(maxmodes):
    plot(freqlist*1e-6, a_s[:,i], 'r--') # antisymmetric
    plot(freqlist*1e-6, s_s[:,i], 'b--') # symmetric

xlim([0, frmax*1e-6])
if NONDIM:
    xlabel(r'$f\cdot h$, MHz$\cdot$mm')
    title(f'Dispersion curves')
else:
    xlabel(r'$f$, MHz')
    title(f'Dispersion curves, h={h*1e3} mm')
ylim([0, 10000]) # max. phase speed
ylabel('$c_p$, m/s')
grid()
show()




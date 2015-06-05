from xml.etree.ElementTree import ElementTree
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy import stats
from scipy.stats import norm
from math import exp,sqrt,pi
from matplotlib import pylab as plt

class ImportTracks:
    
    def __init__(self,path,source='trackmate'):
        
        self.path = path
        if 'trackmate' in source:
            self.root = self.get_root()
            self.date_time = self.root.attrib['generationDateTime']
            self.software = self.root.attrib['from']
            self.frame_interval = float(self.root.attrib['frameInterval'])
            self.space_units = self.root.attrib['spaceUnits']
            self.time_units = self.root.attrib['timeUnits']
            self.ntracks = int(self.root.attrib['nTracks'])
            self.trajectories = self.trackmate_tracks()
        elif 'palmtracer' in source:
            self.software = 'palmtracer'
            self.ntracks = None
            self.trajectories = self.palmtracer_tracks()
        
    def get_root(self):
        return ElementTree(file=self.path).getroot()
    
    def trackmate_tracks(self):
        
        df_list = []
        for particle,track_node in enumerate(self.root):
            
            nspots = int(track_node.attrib['nSpots'])
            
            if len(track_node.getchildren()) != nspots:
                nspots = len(track_node.getchildren())
                
            track = np.zeros((nspots,6))
            
            for i,detection_node in enumerate(track_node):
                
                track[i,0] = float(detection_node.attrib['t'])
                track[i,1] = particle+1
                track[i,2] = track[i,0] * float(self.frame_interval)
                track[i,3] = float(detection_node.attrib['x'])
                track[i,4] = float(detection_node.attrib['y'])
                track[i,5] = float(detection_node.attrib['z'])
                
            df_list.append(pd.DataFrame(track,columns=['frame','particle','t','x','y','z']))
                
        return pd.concat(df_list).reset_index(drop=True)
    
    def kunle_palmtracer_tracks(self):
        num_lines = sum(1 for line in open(self.path))
        try:
            with open(self.path) as t_in:
                data = pd.read_csv(t_in,header=None,\
                                   sep='\t',engine='c',\
                                   skiprows=range(num_lines-50,num_lines),\
                                   index_col=False,low_memory=False)  

            data.columns = ['particle','frame','x','y',\
                       'good','intensity','extra1','extra2','t']
            return data
        except:
            print 'there was a problem parsing localisation data'
            return None
        
    def palmtracer_tracks(self):
        num_lines = sum(1 for line in open(self.path))
        try:
            with open(self.path) as t_in:
                data = pd.read_csv(t_in,header=None,\
                                   sep='\t',engine='c',\
                                   skiprows=range(num_lines-50,num_lines),\
                                   index_col=False,low_memory=False)  

            data.columns = ['particle','frame','x','y',\
                       'good','intensity']
            return data
        except:
            print 'there was a problem parsing localisation data'
            return None


"""
    Borrowed from trackpy - msd,imsd,emsd and fit_powerlaw
"""

def fit_powerlaw(data, **kwargs):
    """Fit a powerlaw by doing a linear regression in log space."""
    ys = pd.DataFrame(data)
    values = pd.DataFrame(index=['slope', 'intercept'])
    fits = {}
    for col in ys:
        y = ys[col].dropna()
        x = pd.Series(y.index.values, index=y.index, dtype=np.float64)
        slope, intercept, r, p, stderr = \
            stats.linregress(np.log(x), np.log(y))
        values[col] = [slope, np.exp(intercept)]
        fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope)
    values = values.T
    fits = pd.concat(fits, axis=1)
    return (values,fits)

def linear_regress(data, log=True, clip=None, r2=0.8, **kwargs):
    """Fit a 1st order polynomial by doing first order polynomial fit."""
    ys = pd.DataFrame(data)
    values = pd.DataFrame(index=['slope', 'intercept', 'good'])
    good = False
    fits = {}
    for col in ys:
        if clip:
            y = ys[col].dropna()
            limit = np.arange(1,np.min(((1+clip),len(y.index))))
            y = ys.loc[limit,[col]][col]
            x = pd.Series(y.index.values, index=y.index, dtype=np.float64)
        else:
            y = ys[col].dropna()
            x = pd.Series(y.index.values, index=y.index, dtype=np.float64)
        if log:
            slope, intercept, r, p, stderr = \
                    stats.linregress(np.log(x), np.log(y))
            if r**2 > r2:
                good = True
            values[col] = [slope, np.exp(intercept), good]
            fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope)
        else:
            slope, intercept, r, p, stderr = \
                    stats.linregress(x, y)
            if r**2 > r2:
                good = True
            values[col] = [slope, intercept, good]
            fits[col] = x.apply(lambda x: intercept*x**slope)
    values = values.T
    fits = pd.concat(fits, axis=1)
    return (values,fits)

def mss(traj, mpp, fps, mu, max_lagtime=100, detail=False):
    """Compute the displacement and moment of one
    trajectory over a range of time intervals.

    Parameters
    ----------
    traj : DataFrame with one trajectory, including columns frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    mu  : exponent
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : See below. Default False.

    Returns
    -------
    DataFrame([<x>, <y>, <x^mu>, <y^mu>, msd], index=t)

    If detail is True, the DataFrame also contains a column N,
    the estimated number of statistically independent measurements
    that comprise the result at each lagtime.

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.

    See also
    --------
    imsd() and emsd()
    """

    pos = traj.set_index('frame')[['x', 'y']]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    max_lagtime = min(max_lagtime, len(t)) # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lagt', 'frames']).abs()
    results = mpp*disp.mean(level=0)
    results.columns = ['<x>', '<y>']
    results[['<x^mu>', '<y^mu>']] = mpp**mu*(disp**mu).mean(level=0)
    results['moment'] = mpp**mu*(disp**mu).mean(level=0).sum(1) # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.icol(0).count(level=0).div(Series(lagtimes))
    if isinstance(fps,int):
        fps = float(fps)
    results['lagt'] = results.index.values/fps
    return results

def emsd(traj, mpp, fps, max_lagtime=100, detail=False):
    """Compute the mean squared displacements of an ensemble of particles.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, True))
        ids.append(pid)
    msds = pd.concat(msds, keys=ids, names=['particle', 'frame'])
    results = msds.mul(msds['N'], axis=0).mean(level=1) # weighted average
    results = results.div(msds['N'].mean(level=1), axis=0) # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    if not detail:
        return results.set_index('lagt')['msd']
    return results

def msd(traj, mpp, fps, max_lagtime=100, detail=False):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals.

    Parameters
    ----------
    traj : DataFrame with one trajectory, including columns frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : See below. Default False.

    Returns
    -------
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    If detail is True, the DataFrame also contains a column N,
    the estimated number of statistically independent measurements
    that comprise the result at each lagtime.

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.

    See also
    --------
    imsd() and emsd()
    """
    pos = traj.set_index('frame')[['x', 'y']]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    max_lagtime = min(max_lagtime, len(t)) # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lagt', 'frames'])
    results = mpp*disp.mean(level=0)
    results.columns = ['<x>', '<y>']
    results[['<x^2>', '<y^2>']] = mpp**2*(disp**2).mean(level=0)
    results['msd'] = mpp**2*(disp**2).mean(level=0).sum(1) # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.icol(0).count(level=0).div(Series(lagtimes))
    results['lagt'] = results.index.values/fps
    return results

def imsd(traj, mpp, fps, max_lagtime=100, statistic='msd', mu=2):
    """Compute the mean squared displacements of particles individually.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    statistic : {'msd', '<x>', '<y>', '<x^2>', '<y^2>'}, default is 'msd'
        The functions msd() and emsd() return all these as columns. For
        imsd() you have to pick one.

    Returns
    -------
    DataFrame([Probe 1 msd, Probe 2 msd, ...], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []

    for pid, ptraj in traj.groupby('particle'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, False))
        ids.append(pid)
    results = pd.concat(msds, keys=ids)
    # Swap MultiIndex levels so that unstack() makes particles into columns.
    results = results.swaplevel(0, 1)[statistic].unstack()
    lagt = results.index.values.astype('float64')/float(fps)
    results.set_index(lagt, inplace=True)
    results.index.name = 'lag time [s]'
    return results

def fit_smss(gamma, mu, r2=0.8):
    slope, intercept, r, p, stderr = \
        stats.linregress(mu, gamma) 
    good = False
    if (slope > 0) and (r**2 > 0.8):
        good = True
    return (slope,good)

def imss(traj, mpp, fps, max_lagtime=100, powerfit=True, clip=None,\
         r2=0.8):
    smss = []
    D = []
    good = []
    gamma_list = []
    for pid,ptraj in traj.groupby('particle'):
        all_mu = np.arange(7)
        gamma = np.zeros(7)
        for mu in all_mu:
            # get all the moments including the msd (mu=2)
            moment = mss(ptraj, mpp, fps, mu, max_lagtime, False)
            moment.set_index('lagt',inplace=True)
            vals,fits = linear_regress(moment['moment'],log=powerfit,\
                                      clip=clip,r2=r2)
            gamma[mu] = vals['slope'].values[0]
            if mu == 2:
                D.append(0.25*vals['intercept'].values[0])
        # curve fit the MSS
        gamma_fit = fit_smss(gamma,all_mu)
        gamma_list.append(gamma)
        smss.append(gamma_fit[0])   
        good.append(gamma_fit[1])
         
    mss_data = pd.DataFrame()
    mss_data['gamma'] = gamma_list
    mss_data['smss'] = smss
    mss_data['D2'] = D
    mss_data['good'] = good   
    return mss_data   

def smss(ptraj, mpp, fps, max_lagtime, powerfit, clip, r2):
    all_mu = np.arange(7)
    gamma = np.zeros(7)
    for mu in all_mu:
        # get all the moments including the msd (mu=2)
        moment = mss(ptraj, mpp, fps, mu, max_lagtime, False)
        moment.set_index('lagt',inplace=True)
        vals,fits = linear_regress(moment['moment'],log=True)
        gamma[mu] = vals['slope'].values[0]
        if mu == 2:
            D = 0.25*vals['intercept'].values[0]
    # curve fit the MSS
    print gamma
    return (gamma,fit_smss(gamma,all_mu),D)

def segment(trajs,trackid,fmin=0,fmax=100):
    track = trajs[trajs["particle"] == trackid]
    track_seg = track[(track["frame"] >= fmin) & (track["frame"] <= fmax)]
    return track_seg

def single_track(trajs,idx):
    return trajs[trajs["particle"] == idx]

def subset(trajs,start=0,stop=100):
    traj_list = []
    for t in range(start,stop):
        traj_list.append(trajs[trajs["particle"] == t+1])
    return pd.concat(traj_list).reset_index(drop=True)

def filter_by_duration(trajs,duration=10):
    gtrajs = trajs.groupby("particle")
    return gtrajs.filter(lambda x: len(x) > duration)   

def trajs_to_list(trajs):
    traj_list = []
    for pid,ptraj in trajs.groupby("particle"):
        traj_list.append(ptraj)
    return traj_list

def track_lengths(trajs):
    lengths = []
    ntracks = int(trajs["particle"].max())
    g = trajs.groupby("particle")
    for i in g.size():
        lengths.append(i)
    return lengths      

def diffusion_coefficients(results):
    if isinstance(results,list):
        return [results[i][2] \
            for i in range(len(results)) \
            if results[i][1][1]]
    else:
        return results['D2'].values
    
def smss_values(results):
    if isinstance(results,list):
        return [results[j][1][0] \
                for j in range(len(results)) \
                if results[j][1][1]]
    else:
        return results['smss'].values
"""
from the scipy cookbook
"""

def brownian(x0, n, dt, D, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, sqrt(2 * D * dt); 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, sqrt(2 * D * dt); t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    D : float
        D determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is 4*D**2*dt).
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=sqrt(2 * D * dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def brownian_naive(x0, n, dt, D, dims=2, out=None):
    
    k = sqrt(D * dims * dt)

    x0 = np.asarray(x0)
    
    r = k*np.random.randn(2, n)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def confined_naive(x0, n, dt, D, radius=2, dims=2, out=None):
    
    theta = np.linspace(0,2*pi,360)
    
    domainx = radius * np.cos(theta)
    domainy = radius * np.sin(theta)
    
    k = sqrt(D * dims * dt)
    rr = np.zeros((2,n))
    r = np.zeros(2)
    oldr = r
    for i in range(n):
        # take a step
        dr = k*np.random.randn(2)
        # update r
        r += dr
        to_centre = sqrt(r[0]**2+r[1]**2)
        # check that we are inside the domain
        if (to_centre >= radius) or (to_centre >= radius):
            outside = 1
            # if we are outside loop until we are back inside
            while outside:
                # set a new step
                dr = k*np.random.randn(2)
                # update r - add step to oldr (the last one that was inside)
                r = oldr + dr
                to_centre = sqrt(r[0]**2+r[1]**2)
                # check if we are inside
                if (to_centre <= radius) and (to_centre <= radius):
                    # break the loop
                    outside = 0
                    # store the value of r
                    rr[:,i] = r[:]
        else: # we are inside
            # store the value of r
            rr[:,i] = r[:]
        # update oldr - should be inside
        oldr = r
        
                
    x0 = np.asarray(x0)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(rr.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
#     np.cumsum(rr, axis=-1, out=out)
    out = rr
    
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

if __name__ == '__main__':
    N = 1000
    dt = 0.05
    D = 0.1
    dims = 2
    x = np.empty((2,N+1))
    x[:, 0] = 0.0
    cn = confined_naive(x[:,0], N, dt, D, radius=1, dims=2)
    plt.plot(cn[0,:],cn[1,:],'r-')
    plt.plot(cn[0,:],cn[1,:],'ko')
    plt.show()

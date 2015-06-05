from math import pi,sin, floor
import numpy as np
import pandas as pd
from track_analysis import brownian

def export(tracks):

    tracks_df = pd.concat(tracks).reset_index(drop=True)
    file_name = 'simulated_tracks.csv' 
    with file(file_name, 'w') as outfile:
        tracks_df.to_csv(outfile,sep='\t',float_format='%8.2f',index=False,\
                         encoding='utf-8')     

def circ_rand(n,R):
    a = 0
    b = pi
    p = (2*a-sin(2*a)) + (2*b-sin(2*b) - 2*a+sin(2*a)) * np.random.rand(n,1)
    p = np.mod(p+pi,2*pi) - pi
    p = np.sign(p) * np.absolute(p)**(1.0/3.0)

    c1 = 1.0/(6**(1.0/3.0)) # c1 and c2 used in approx. of derivative
    c2 = (c1-2.0/3.0/pi**(2.0/3.0))/pi**2
    t = np.zeros((n,1))
    for k in range(12): # Twelve trips should suffice for convergence
        p1 = t - np.sin(t)
        p1 = np.sign(p1)*np.absolute(p1)**(1.0/3.0)
        t -= (p1-p)/(c1-c2*t**2) # Pseudo Newton Raphson
    
    t = np.mod(t,2*pi)/2.0
    s = -1+2*np.random.rand(n,1)
    x = R * np.cos(t) # x and y are cartesian coordinates of the random points
    y = R * s * np.sin(t)
    return [x,y]

def generate_tracks(clusters, N=500, fps=25.0, mpp=1, D=0.01):
    
    dt = 1/float(fps)
    x = np.empty((2,N+1))
    p_count = 0
    tracks = []
    for seeds in clusters:
        # Initial values of x.
        num_particles = seeds.shape[1]
        particles = []
        for p in range(num_particles):
            x[:, 0] = seeds[:,p]
            bx = brownian(x[:,0], N, dt, D)
            bdf = pd.DataFrame()            
            bdf["particle"] = np.ones(N)*(p_count+1)
            bdf["frame"] = np.arange(N)
            bdf["x"] = bx[0,:]
            bdf["y"] = bx[1,:]
            particles.append(bdf)    
            p_count += 1
        tracks.append(pd.concat(particles).reset_index(drop=True))
    return tracks
    
def generate_seeds(size=(128,128),density=1e-2,num_clusters=25,cluster_radius=25,psf=2):
    
    if (type(size) != tuple) or (len(size) != 2):
        return None

    N = density*size[0]*size[1]
    P = int(floor(N/float(num_clusters)))

    # define some random background points
    bg =  np.array((np.random.uniform(0,size[0],100),\
                    np.random.uniform(0,size[1],100)))
                    
    # define the cluster seeds
    seeds =  np.array((np.random.uniform(0,size[0],num_clusters),\
                np.random.uniform(0,size[1],num_clusters)))
                
    # define the clusters
    clusters = []
    for i in range(num_clusters):
        r = circ_rand(P,cluster_radius)
        rx = r[0] + seeds[0,i]
        ry = r[1] + seeds[1,i]
        x = rx[:,0]
        y = ry[:,0]
        x = x[(x > 0.0) & (x < size[0])]
        y = y[(y > 0.0) & (y < size[1])]
        coords = np.array((x,y))
        if len(coords.shape) == 2:
            clusters.append(coords)
            
    return bg,clusters

if __name__=='__main__':
    bg,clusters = generate_seeds(density=0.01,cluster_radius=4)
#Gerald Schuller, 2011
#Translated from Matlab/Octave
#Gerald Schuller, June 2022

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def fbinterp(fb,n,N):
   #interpolates Low Delay filter bank coefficients for a higher
   #number of bands, by a factor of n
   #usage: fn=fbinterp(fb,n,N);
   #N: number of bands of fb
   #fb: old fb coeff.
   #n: factor with which the size of fb is multiplied
   #m: order of interpolating polynomial, recommended: m=3,  
   #m+1<=N/2, N is the number of bands of fb 
   
   xn=(np.arange((n*N/2))+0.5)/n-0.5;
   print("xn=", xn)
   L=max(fb.shape);
   blocks=L//(N//2);
   fn=[]
   for i in range(blocks):
     x=np.arange(N/2)
     y=fb[i*N//2+np.arange(N//2)]
     f = CubicSpline(x, y)
     fn=np.hstack((fn,f(xn)))

   return fn

#Testing:   
if __name__ == '__main__':
   import matplotlib.pyplot as plt
   
   fbsym=np.loadtxt('fbsy256t127d64bbitb.mat')
   plt.plot(fbsym)
   plt.title("Test LDFB coefficient set")
   
   fn=fbinterp(fb=fbsym,n=0.5,N=64);
   plt.figure()
   plt.plot(fn)
   plt.title("Interpolated coefficients")
   plt.show()
   

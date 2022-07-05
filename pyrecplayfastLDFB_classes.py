# coding: utf-8 
#Program for the fast implementation of the MDCT and Low Delay analysis and synthesis filter bank.
#Gerald Schuller, Nov. 2014.
#Algorithm according to:
#G. Schuller and T. Karp: "Modulated Filter Banks with Arbitrary System Delay: Efficient Implementations and #the Time-Varying Case", IEEE Transactions on Signal Processing, March 2000, pp. 737–748 
#updated July 2016
#Implementaion using classes, to make it usuable for stereo and multichannel parallel processing.
#Gerald Schuller, June 2022

import numpy as np
import scipy.fftpack as spfft

#The Low Delay Filter Bank:------------------------------

#classes:

class Dmatrixclass:

   def __init__(self, N):
      self.z=np.zeros(N//2)
      self.zi=np.zeros(N//2)
      self.N=N
      
   def forward(self,samples):
      out=np.zeros(self.N)
      out[0:(self.N//2)]=self.z
      self.z=samples[0:(self.N//2)]
      out[self.N//2:self.N]=samples[self.N//2:self.N]
      return out

   def inverse(self,samples):
      #Delay elements:
      N=len(samples)
      out=np.zeros(N)
      out[(self.N//2):self.N]=self.zi
      self.zi=samples[(N//2):self.N]
      out[0:self.N//2]=samples[0:self.N//2]
      return out


class Gmatrixclass:

   def __init__(self, N, ecoeff):
      self.z=np.zeros(N//2)
      self.zi=np.zeros(N//2)
      self.N=N
      self.ecoeff=ecoeff
      
   def forward(self,samples):
      #implementation of the delay matrix G(z)
      #N=len(samples)
      #Anti-diagonal ones, flip input:
      out=samples[::-1]
      #Delay elements and coeff in the upper half of the diagonal:
      out[0:(self.N//2)]=out[0:(self.N//2)]+self.z * self.ecoeff
      self.z=samples[0:(self.N//2)]
      return out
      
   def inverse(self,samples):
      #implementation of the zero delay matrix G(z)
      #N=len(samples)
      #Anti-diagonal ones, flip input:
      out=samples[::-1]
      #Delay elements and flipped, neg. coeff in the lower half of the diagonal:
      out[(self.N//2):self.N]=out[(self.N//2):self.N] - self.zi * self.ecoeff[::-1]
      self.zi=samples[(self.N//2):self.N]
      return out

class H2matrixclass:

   def __init__(self, N, H2coeff):
      self.z=np.zeros(N//2)
      self.zi=np.zeros(N//2)
      self.N=N
      self.ecoeff=ecoeff
      
   def forward(self,samples):
      #implementation of the maximum delay matrix H(z)
      #N=len(samples)
      #Anti-diagonal delays, flip delayed input:
      out=self.z[::-1]
      #input mult. with coeff in the upper half of the diagonal:
      out[0:(self.N//2)]=out[0:(self.N//2)]+ samples[0:(self.N//2)]* H2coeff
      self.z=samples
      return out
      
   def inverse(self,samples):
      #N=len(samples)
      #Anti-diagonal delays, flip delayed input:
      out=self.zi[::-1]
      #input mult. with neg. flipped coeff in the lower half of the diagonal:
      
      out[(self.N//2):self.N]=out[(self.N//2):self.N]- samples[(self.N//2):self.N]* H2coeff[::-1]
      self.zi=samples
      return out

class LDFBclass: #A Low Delay filter bank with an F and two G folding matrices:

   def __init__(self, N, ldfbcoeff):
      self.N=N
      self.ldfbcoeff=ldfbcoeff
      self.Dmatrix=Dmatrixclass(N=N)
      #G_0 matrix:
      g0coeff=ldfbcoeff[int(1.5*N):(2*N)]
      self.G0matrix=Gmatrixclass(N, g0coeff)
      #G1 matrix:
      g1coeff=ldfbcoeff[(2*N):int(2.5*N)]
      self.G1matrix=Gmatrixclass(N, g1coeff)
   
   def forward(self, samples):
      #N=len(samples)
      #load LDFB coefficients:
      #Fmatrix:
      fcoeff=self.ldfbcoeff[0:int(1.5*N)]
      y=symFmatrix(samples, fcoeff)
      y=self.Dmatrix.forward(y)
      y=self.G0matrix.forward(y)
      y=self.G1matrix.forward(y)
      y=DCT4(y)
      return y
      
   def inverse(self, samples):
      #y: N subband samples or frequency coefficients
      #N=len(y)
      
      #inverse DCT4 (basically identical to DCT4):
      x=iDCT4(samples)
      #Zero-Delay matrices:
      x=self.G1matrix.inverse(x)
      x=self.G0matrix.inverse(x)
      #inverse D(z) matrix:
      x=self.Dmatrix.inverse(x)
      #inverse F matrix:
      #load LDFB coefficients:
      fcoeff=self.ldfbcoeff[0:int(1.5*N)]
      x=symFinvmatrix(x,fcoeff)
      return x

class MDCTclass: # An MDCT filter bank with an F folding matrix:

   def __init__(self, N, mdctcoeff):
      self.N=N
      self.mdctcoeff=mdctcoeff
      self.Dmatrix=Dmatrixclass(N=N)
   
   def forward(self, samples):
      #N=len(samples)
      #load LDFB coefficients:
      #fcoeff=mdctcoeff[0:int(1.5*self.N)]
      fcoeff=self.mdctcoeff
      #Fmatrix:
      y=symFmatrix(samples, fcoeff)
      y=self.Dmatrix.forward(y)
      y=DCT4(y)
      return y
      
   def inverse(self, samples):
      #y: N subband samples or frequency coefficients
      #N=len(y)
      #load LDFB coefficients:
      #fcoeff=fb[0:int(1.5*self.N)]
      fcoeff=self.mdctcoeff
      
      #inverse DCT4 (is basically identical to DCT4):
      x=iDCT4(samples)
      #inverse D(z) matrix:
      x=self.Dmatrix.inverse(x)
      #inverse F matrix:
      x=symFinvmatrix(x,fcoeff)
      return x


#The symmetric F Matrix function:
#implements multiplication samples*symFmatrix 
def symFmatrix(samples,fcoeff):
   sym=1.0;
   N=len(samples)
   out=np.zeros(N)
   out[0:(N//2)]=(fcoeff[0:(N//2)]*samples[0:(N//2)])[::-1] +fcoeff[(N//2):N]*samples[(N//2):N]
   out[(N//2):N]=(fcoeff[N:(N+N//2)]*samples[0:(N//2)]) 
   #+-1=det([a,b;c,d]) =a*d-b*c => d=(+-1+b*c)/a:
   ff= (-sym*np.ones(N//2)+ fcoeff[N:int(1.5*N)][::-1]*fcoeff[(N//2):N])/fcoeff[0:(N//2)][::-1]
   out[(N//2):N]= out[(N//2):N] +(ff*samples[(N//2):N])[::-1]
   return out

def symFinvmatrix(samples,fcoeff):
   #inverse symFamtrix, uses order for the synthesis F matrix as shown in Audio Coding lecture FB2,
   #but with coefficients in reverse order, since the lecture has h as a window and not impulse response.
   #That is also why the negative signa has to be moved from the end to the beginning:
   sym=1.0
   N=len(samples)
   out=np.zeros(N)
   ff= (-sym*np.ones(N//2)+ fcoeff[N:int(1.5*N)][::-1]*fcoeff[(N//2):N])/fcoeff[0:(N//2)][::-1]

   out[0:(N//2)]=-ff[::-1]*(samples[0:(N//2)][::-1])+fcoeff[int(0.5*N):N][::-1]*samples[(N//2):N]
   out[(N//2):N]=fcoeff[N:int(1.5*N)][::-1]*samples[0:(N//2)]- fcoeff[0:int(0.5*N)][::-1]*(samples[(N//2):N][::-1])
   return out

#The DCT4 transform:
"""
def DCT4(samples):
   #use a DCT3 to implement a DCT4:
   N=len(samples)
   samplesup=np.zeros(2*N)
   #upsample signal:
   samplesup[1::2]=samples
   y=spfft.dct(samplesup,type=3)/2
   return y[0:N]
"""
def DCT4(samples):
   #use a DCT4, now native in Python3 
   y=spfft.dct(samples,type=4, norm='ortho')
   return y
   
def iDCT4(samples):
   #use a DCT4, now native in Python3 
   y=spfft.idct(samples,type=4, norm='ortho')
   return y

if __name__ == '__main__':
   #To avoid error "no Qt platform plugin could be initialized":
   #(Incompatibilites between matplotlib and opencv-python)
   #uncomment if necessary:
   """
   import sys
   from PyQt5 import QtWidgets
   QtWidgets.QApplication(sys.argv)
   """
   #For visualizations:
   import matplotlib.pyplot as plt
   #import plotext as plt  #alternative plotting in terminal
   #plotting frequency responses:
   from scipy.signal import freqz
   import time
   import cv2
   #For real time audio:
   import pyaudio
   import struct

   #---------------------------
   import fbinterp2 #for generating coefficients for other number of subbands N
   #Test Filter Bank:

   #Runs real time audio processing:----------------------------

   #N=64
   N=512 #Number of subbands and block size, has to be even. 

   #Load the LDFB coefficients from text file:
   if N==64:
     #256 taps, 127 delay, 64 subbands:
     fbsym=np.loadtxt('fbsy256t127d64bbitb.mat')
   elif N==512:
     #2048 taps, 1023 delay, 512 subbands:
     fb=np.loadtxt('fb2048t1023d512bbitcs.mat')
     #We assume a symmetric F matrix (det=1), hence we
     #only need the first 1.N coefficients of F, not 2N:
     fbsym=np.append(fb[:int(1.5*N)],fb[2*N:])
     #fb=fbsym
   else:
     fb=np.loadtxt('fb2048t1023d512bbitcs.mat')
     fbsym=np.append(fb[:int(1.5*512)],fb[2*512:])
     n=N/512
     fbsym=fbinterp2.fbinterp(fb=fbsym,n=n,N=N)
     
     
   print("fbsym=", fbsym)
   #example: Sine window:
   #fb=np.sin(np.pi/(2*N)*(np.arange(0,int(1.5*N))+0.5))

   display=True

   #initialize filter bank memory:
   #initFB(N)
   #Instanciate low delay filer bank, forward/analysis and inverse/synthesis:
   LDFB=LDFBclass(N, fbsym)
   #get and plot the synthesis impulse response of subband 0: 
   y=np.ones((N,10))*0.0;
   y[0,0]=1.0
   xrek=np.ones(10*N)*0.0
   for m in range(10):
     #print("y[:,m]=", y[:,m])
     xrek[(m*N):((m+1)*N)]=LDFB.inverse(y[:,m])
   #Plots synthesis impulse response of subband 0:
   #"""
   plt.plot(xrek)
   plt.title('LDFB Impulse Response of Subband 0 of the Synthesis Low Delay FB')
   w,H=freqz(xrek,worN=2048)
   plt.figure()
   plt.plot(w,20*np.log10(np.abs(H)+1e-6))
   #Enlarge normalized frequencies in range of 0 to 0.1:
   #plt.axis([0, 0.1, -60,5])
   plt.title('Its Enlarged Magnitude Frequency Response') 
   plt.ylabel('dB Attenuation')
   plt.xlabel('Normalized Frequency (pi is Nyquist Freq.)')
   plt.show()
   #"""

   CHUNK = N #Blocksize for the sound card
   WIDTH = 2 #2 bytes per sample
   CHANNELS = 1 #2
   RATE = 32000  #Sampling Rate in Hz
   p = pyaudio.PyAudio()

   a = p.get_device_count()
   print("device count=",a)

   for i in range(0, a):
      print("i = ",i)
      b = p.get_device_info_by_index(i)['maxInputChannels']
      print(b)
      b = p.get_device_info_by_index(i)['defaultSampleRate']
      print(b)
   #open sound device:
   stream = p.open(format=p.get_format_from_width(WIDTH),
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  output=True,
                  #input_device_index=3,
                  frames_per_buffer=CHUNK)


   print("* recording")
   #Waterfall diagram:
   #Size of waterfall diagramm:
   #N cols:
   rows=500
   cols=N

   frame=0.0*np.ones((rows,cols,3));
   ctr=0
   while(True):
      ctr=ctr+1
      #Reading from audio input stream into data with block length "CHUNK":
      data = stream.read(CHUNK)
      #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
      #shorts = (struct.unpack( "128h", data ))
      shorts = (struct.unpack( 'h' * CHUNK, data ));
      samples=np.array(list(shorts),dtype=float);
      if np.max(np.abs(samples))>32000:
              print("Overload input")

      #shift "frame" 1 up:
      if (ctr%16 ==0):
         frame[0:(rows-1),:]=frame[1:rows,:]; 
     
      #This is the analysis MDCT of the input: 
      y=LDFB.forward(samples[0:N])

      #yfilt is the processed subbands, processing goes here:
      yfilt=y
      #yfilt=np.zeros(N)
      #yfilt[10:150]=y[10:150]
      #yfilt[30]=y[30]*4
      #yfilt[0:1024]=y[0:1024]
      
      #Waterfall color mapping when loop counter ctr reaches certain values:
      if ((display==True) & (ctr%16 ==0)):
         R=0.25*np.log((np.abs(yfilt/np.sqrt(N))+1))/np.log(10.0)
         #Red frame:
         frame[rows-1,:,2]=R
         #Green frame:
         frame[rows-1,:,1]=np.abs(1-2*R)
         #Blue frame:
         frame[rows-1,:,0]=1.0-R
         #frame[rows-1,:,0]=frame[rows-1,:,1]**3
         # Display the resulting frame
         cv2.imshow('MDCT Waterfall (end with "q")',frame)

      #Inverse, Synthesis filter bank:
      #Inverse/synthesis MDCT:
      xrek=LDFB.inverse(yfilt);
      if np.max(np.abs(xrek))>32000:
           print("Overload output")
      xrek=np.clip(xrek, -32000, 32000)
      xrek=xrek.astype(int)
      #converting from short integers to a stream of bytes in "data":
      #data=struct.pack('h' * len(samples), *samples);
      data=struct.pack('h' * len(xrek), *xrek);
      #Writing data back to audio output stream: 
      stream.write(data, CHUNK)

      #Keep window open until key 'q' is pressed:
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
   # When everything done, release the capture

   cv2.destroyAllWindows()

   stream.stop_stream()
   stream.close()
   p.terminate()


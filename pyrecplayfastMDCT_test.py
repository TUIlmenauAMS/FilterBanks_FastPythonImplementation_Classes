# coding: utf-8 
#Program for testing the fast classes implementation of the Low Delay filter bank (LDFB), for analysis and synthesis,
#using it to implement a Modified Discrete Cosine Transform (MDCT) filter bank as a special case of an LDFB.
#for real time audio streming.
#June 2022

import numpy as np
import scipy.fftpack as spfft

from pyrecplayfastLDFB_classes import *



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
   import cv2
   import matplotlib.pyplot as plt
   #import plotext as plt  #alternative plotting in terminal
   #plotting frequency responses:
   from scipy.signal import freqz
   import time
   
   #For real time audio:
   import pyaudio
   import struct
   
   #---------------------------
   #Test Filter Bank:
   
   #Runs real time audio processing:----------------------------

   N=512 #Number of subbands and block size, works for all even N
   #example: Sine window:
   fb=np.sin(np.pi/(2*N)*(np.arange(0,int(1.5*N))+0.5))
   
   #Instantiate MDCT (forward and inverse):
   MDCT=MDCTclass(N,mdctcoeff=fb)

   display=True

   #get and plot the synthesis impulse response of subband 0:
   y=np.ones((N,10))*0.0;
   y[0,0]=1.0
   xrek=np.ones(10*N)*0.0
   for m in range(10):
      xrek[(m*N):((m+1)*N)]=MDCT.inverse(y[:,m])
   #Plots synthesis impulse response of subband 0:
   print("xrek=", xrek)
   #"""
   plt.plot(xrek)
   plt.title('MDCT Impulse Response of Subband 0 of the Synthesis FB')
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
   #Real time audio from and to sound card:
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
       y=MDCT.forward(samples[0:N])

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
       xrek=MDCT.inverse(yfilt);
       if np.max(np.abs(xrek))>32000:
          print("Overload output")
       xrek=np.clip(xrek, -32000, 32000)

       #converting from short integers to a stream of bytes in "data":
       #data=struct.pack('h' * len(samples), *samples);
       #data=struct.pack('h' * len(xrek), *xrek);
       data = xrek.astype(np.int16).tobytes()
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


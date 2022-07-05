"""
Live multichannel audio for source separation, using the callback function.
Gerald Schuller, 2022, February
"""

import pyaudio
import struct
#import math
import time
#import array
import numpy as np
import scipy
import tkinter as Tk 
from pyrecplayfastLDFB_classes import *

N=512 #Number of subbands and block size, works for all even N
#example: Sine window:
fb=np.sin(np.pi/(2*N)*(np.arange(0,int(1.5*N))+0.5))
#Instantiate MDCT (forward and inverse):
#1 MDCT for each channel:
MDCT0=MDCTclass(N,mdctcoeff=fb)
MDCT1=MDCTclass(N,mdctcoeff=fb)

CHUNK = N #Blocksize
#WIDTH = 2 #2 bytes per sample
CHANNELS = 2 #2
RATE = 32000  #Sampling Rate in Hz
RECORD_SECONDS = 8
multichandata=np.zeros((CHUNK,CHANNELS));
xrek=np.zeros((CHUNK,CHANNELS))
volume=0.9

p = pyaudio.PyAudio()
a = p.get_device_count()
print("device count=",a)



def quitbutton():
   #quits the program
   global PLAY
   PLAY=False
   return

def processingbutton():
   #quits the program
   global PROC
   PROC=not PROC
   return


PLAY = True
PROC = True
master = Tk.Tk()
#PROC=Tk.IntVar()
vol = Tk.Scale(master, from_=20, to=-30, length=200, tickinterval=8, label='Volume dBFs')
vol.set(-10) #start
vol.pack()
Tk.Button(master, text='Quit', command=quitbutton).pack()
Tk.Button(master, text='Processing on/off', command=processingbutton).pack()
#Tk.Checkbutton(master, text="Processing", variable=PROC).pack()

for i in range(0, a):
    print("i = ",i)
    b = p.get_device_info_by_index(i)['maxInputChannels']
    print(b)
    b = p.get_device_info_by_index(i)['defaultSampleRate']
    print(b)



def callback(in_data, frame_count, time_info, flag):
    global multichandata, xrek, writing #global variables 
    #print ("frame_count=", frame_count)
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    #de-interleaving:
    multichandata=np.reshape(audio_data, (-1,CHANNELS))
    #Stereo MDCT here:
    y0=MDCT0.forward(multichandata[:,0])
    y1=MDCT1.forward(multichandata[:,1])
    #print("y0.shape=", y0.shape)
    if PROC: #processing active:
       #Example, e.g. for hearing aids: Copy and add frequencies > 10 kHz below, 192 subbands, or 6KHz shift below:
       y0[128:320]+=y0[320:]
       y1[128:320]+=y1[320:]
       #gain of 20 dB above 6 kHz:
       y0[192:]*=10
       y1[192:]*=10
    xrek[:,0]=MDCT0.inverse(y0)*volume;
    xrek[:,1]=MDCT1.inverse(y1)*volume; 
    #Interleaving back:
    audio_data= np.reshape(xrek, (-1,1))[:,0]
    audio_data = audio_data.astype(np.float32).tobytes()
    # fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
    return (audio_data, pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32, #p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                #input_device_index=3,
                frames_per_buffer=CHUNK,
                stream_callback=callback)
                

print("* recording")


stream.start_stream()
n=0;
while stream.is_active() and PLAY==True:
    master.update()
    volume=vol.get() #dBFs
    volume=10**(volume/20)
    #Loop keeps callback running
    #print("active, multichandata.shape=", multichandata.shape)
    #print("multichandata=", multichandata)
    #Possibly optimization here
    n+=1
    if n%10==0:
       print("n=", n, "PROC=", PROC)
    time.sleep(0.1)  


# stop stream 
stream.stop_stream()
stream.close()

print("Done")

p.terminate()


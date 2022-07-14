# FilterBanks_FastPythonImplementation_Classes

These are programs and functions for the implementation of fast filter banks, Low Delay filter banks, and Modified Discrete Cosine Transform filter banks. They use a classes implementation, which makes them more easily suitable for multichannel signal processing. For each channel, an instance of the desired filter bank can be instantiated.

The advantages of the fast implementation are:

* fastest implementation without the need of specialized hardware
* low memory requirement for filtering and signal processing
* real time processing, one signal block goes in, one signal block comes out

For the theoretical background see the book:

Gerald Schuller: "Filter Banks and Audio Coding - Compressing Audio Signals Using Python",
Springer 2020, 
ISBN: 978-3-030-51249-1 (e-book)
ISBN: 978-3-030-51251-4 (softcover or hardcover)

The file "pyrecplayfastLDFB_classes.py" contains the filter bank functions and objects (classes).
If run on its own, with 

``python3 pyrecplayfastLDFB_classes.py``

it executes its main section, which loads the coefficient files fbsy256t127d64bbitb.mat (for N=64 subbands) and fb2048t1023d512bbitcs.mat (for N=512 subbands),
and sets up a Low Delay filter bank with N=512 subbands. N can be chosen, but must be an even number. 
If a number of subbands N is chosen differently from 512 or 64, it suitably interpolates the needed coefficients from the case of N=512 using the 
interpolation function in "fbinterp2.py".

After setting up the filter bank, it accesses the sound card, using pyaudio, for a sound stream and displays the real time output of the Low Delay analysis filter bank as a live color spectrogram, using opencv. It also takes the subband signals, which may optionally be modified there, reconstructs an audio signal using the synthesis filter bank, and plays the sound back through the sound card.

The program "pyrecplayfastMDCT_test.py" import the functions and objects from "pyrecplayfastLDFB_classes.py" to set up an 
MDCT (Modified Discrete Cosine Transform) filter bank as another example, also displays a live audio spectrogram, and reconstructs the audio with the synthesis filter bank.

Run it with: ``python3 pyrecplayfastMDCT_test.py``

The final example, "pyrecplaystereoMDCTcallback_tkinter.py", demonstrates the multichannel live processing capability which is obtained 
using objects and instances. It sets up 2 MDCT filter banks for the 2 channels of a stereo audio input from the local sound card, using pyaudio, uses live processing to generate the MDCT subbands with the MDCT analysis filter bank, with possible subband processing and modifications, reconstructs them using the MDCT synthesis filter bank and sends the processed audio streams to the sound card output. The processing is done in a separate thread using a callback function, for uninterrupted audio streaming. An example of the subband processing can be a binaural hearing aid application (except for the delay). This is shown in the example, where frequencies above 10 kHz are shifted below, and frequencies above 6 kHz are amplified by 20 dB.

Outside of this thread, the audio processing can be controlled using "tkinter", which presents a volume control in dB using a slider, a "Quit" button,  "HF Emph. on/off" and "HF Shift on/off" buttons.

Connect headphones and microphones, and run it with: 
``python3 pyrecplaystereoMDCTcallback_tkinter.py``

Many greetings,
  Gerald Schuller

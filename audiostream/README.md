# Audio Stream Example

A simple example that streams your mic input into the Online Prediction Framework (OPF), 
and outputs a prediction, an anomaly score and a likelihood score, based on how familiar t
he model has become to that particular mic input sequence. Think of it as being able to 
recognize a sound, or become more familiar with your speech pattern, and its ability to 
predict what level is next.

The audio is transformed into the frequency domain using a Fast Fourier Transform (FFT), 
and only one frequency bin is taken as input to the model. Meaning that the amplitud of the 
selected bin (frequency) is streamed into the model and then it starts analyzing that
particular frequency for anomalies and predictions. 

## Requirements

- Mac OS X
- [matplotlib](http://matplotlib.org/)
- [pyaudio](http://people.csail.mit.edu/hubert/pyaudio/)

## Usage

    python audiostream.py

This script will run automatically & forever.
To stop it, use KeyboardInterrupt (CRTL+C).

The model also uses a model_params.py file that includes the
parameters to use in the analysis.  

## General algorithm:

1. Mic input is received (voltages in the time domain)
2. Mic input is transformed into the frequency domain, using fast fourier transform (FFT)
3. A frequency range (bin) is selected
4. That changing bin value is fed into the opf model in every iteration of the script
5. The model computes prediction, anomaly and likelihood values.
6. All 4 values (input, prediction, anomaly and likelihood) are plotted.

## Plot includes:
4 time changing lines corresponding to:

1.  Raw input
2.  Predicted value
3.  Anomaly score
4.  Likelihood

## Next steps:

1. Look into better algorithms to pick out the frequency peaks (sound fingerprinting). This could be application specific, and user can determine how to select frequency bins.

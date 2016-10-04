#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
See README.md for details.
"""

"""
numpy - the language of pyaudio (& everything else)
pyaudio - access to the mic via the soundcard
pyplot - to plot the sound frequencies
bitmaparray - encodes an array of indices into an SDR
TP10X2 - the C++ optimized temporal pooler (TP)
"""
import numpy
import pyaudio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot 					as plt
from collections							import deque
from nupic.data.inference_shifter 			import InferenceShifter 
from nupic.frameworks.opf.modelfactory		import ModelFactory 
from nupic.algorithms.anomaly_likelihood	import AnomalyLikelihood

import model_params

WINDOW = 60

class AudioPrediction:

	def __init__(self):

		"""
		Setup the plot, interactive mode on, title, etc.
		Rescale the y-axis
		"""
		plt.ion()
		fig = plt.figure()
		plt.title('Audio Stream example')
		plt.xlabel('Time')
		plt.ylabel('Frequency Level [dB]')
		yLimit = 200
		plt.ylim(0, yLimit)


		"""
		Create model, set predicted field, and likelihood
		"""
		model 		= ModelFactory.create(model_params.MODEL_PARAMS)
		model.enableInference({'predictedField' : 'binAmplitude'})
		
		likelihoods = AnomalyLikelihood()

		shifter 	= InferenceShifter()

		actHistory 	= deque([0.0] * WINDOW, maxlen = 60)
		predHistory	= deque([0.0] * WINDOW, maxlen = 60)
		anomHistory = deque([0.0] * WINDOW, maxlen = 60)
		likeHistory	= deque([0.0] * WINDOW, maxlen = 60)

		actline, 	= plt.plot(range(WINDOW), actHistory)
		predline, 	= plt.plot(range(WINDOW), predHistory)
		anomline,	= plt.plot(range(WINDOW), anomHistory)
		likeline,	= plt.plot(range(WINDOW), likeHistory)	

		"""
		Instance of the class to stream audio
		"""
		audio = AudioStream()
		while audio.start==False:1


		while True:


			inputLevel	= audio.audioFFT[1]

			# Clip input
			maxLevel = model_params.MODEL_PARAMS['modelParams']['sensorParams']['encoders']['binAmplitude']['maxval'] 
			if inputLevel >  maxLevel:
				inputLevel = maxLevel

			modelInput 	= {'binAmplitude' : inputLevel}
			result 		= shifter.shift(model.run(modelInput))

			inference 	= result.inferences['multiStepBestPredictions'][5]
			anomaly 	= result.inferences['anomalyScore']
			likelihood 	= likelihoods.anomalyProbability(inputLevel, anomaly)

			if anomaly is not None:
				actHistory .append(result.rawInput['binAmplitude'])
				predHistory.append(inference)
				anomHistory.append(anomaly * yLimit/2)
				likeHistory.append(likelihood * yLimit/2)


			actline	.set_ydata(actHistory)
			predline.set_ydata(predHistory)
			anomline.set_ydata(anomHistory)
			likeline.set_ydata(likeHistory)

			plt.draw()
			plt.legend(('actual','predicted', 'anomaly', 'likelihood'))


class AudioStream:

	def __init__(self):

		"""
		Sampling details
		 rate: The sampling rate in Hz of my soundcard
		 buffersize: The size of the array to which we will save audio segments (2^12 = 4096 is very good)
		 secToRecord: The length of each sampling
		 buffersToRecord: how many multiples of buffers are we recording?
		"""
		rate			=44100
		self.bufferSize =2**12
		bitResolution	= 16
		binSize			= int(rate/self.bufferSize)
		self.start		= False


		"""
		Setting up the array that will handle the timeseries of audio data from our input
		"""
		if bitResolution == 8:
			width = 1
			self.audioIn = numpy.empty((self.bufferSize), dtype = "int8")
			print "Using 8 bits"
		if bitResolution == 16:
			width = 2
			self.audioIn = numpy.empty((self.bufferSize), dtype = "int16")
			print "Using 16 bits"
		if bitResolution == 32:
			width = 4
			self.audioIn = numpy.empty((self.bufferSize), dtype = "int32")
			print "Using 32 bits"


		"""
		Creating the audio stream from our mic. This includes callback function for
		non blocking mode. This means the callback executes everytime whenever it needs 
		new audio data (to play) and/or when there is new (recorded) audio data available. 
		Note that PyAudio calls the callback function in a separate thread. 
		"""
		p = pyaudio.PyAudio()

		def callback(in_data, frame_count, time_info, status):
			"""
			Replaces processAudio()
			"""
			self.audioIn 	= numpy.fromstring(in_data, dtype = numpy.int16)
			self.audioFFT	= self.fft(self.audioIn)
			# Get the frequency levels in dBs 
			self.audioFFT 	= 20*numpy.log10(self.audioFFT)
			self.start		= True
			return (self.audioFFT, pyaudio.paContinue)

		self.inStream = p.open(format 	=p.get_format_from_width(width, unsigned = False),
							channels	=1,
							rate		=rate,
							input 		=True,
							frames_per_buffer= self.bufferSize,
							stream_callback  = callback)


		"""
		Print out the inputs
		"""
		print "Sampling rate (Hz):\t" + str(rate)
		print "Bit Depth:\t\t"  + str(bitResolution)
		print "Buffersize:\t\t" + str(self.bufferSize)


		


	def fft(self, audio):

		"""
		Fast Fourier Transform - 

		Output:
		'output' - the transform of the audio input into frequency domain. 
		Contains the strength of each frequency in the audio signal
		frequencies are marked by its position in 'output':
		frequency = index * rate / buffesize
		output.size = buffersize/2 
		Use only first half of vector since the second is repeated due to 
		symmetry.

		Great info here: http://stackoverflow.com/questions/4364823/how-to-get-frequency-from-fft-result
		"""
		output = numpy.abs(numpy.fft.fft(audio))
		return output [0:int(self.bufferSize/2)]



audiostream = AudioPrediction()

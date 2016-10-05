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
Example of audio stream to compute predictions, anomaly and likelihood 
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

class AudioPrediction:

	def __init__(self):

		"""
		Setup the plot, interactive mode on, title, and ylimits
		"""
		plt.ion()
		fig = plt.figure()
		plt.title('Audio Stream example')
		plt.xlabel('Time')
		plt.ylabel('Frequency Level [dB]')
		yLimit = 200
		xLimit = 60
		plt.ylim(0, yLimit)


		"""
		Create model, set predicted field, likelihoods and shifter
		"""
		model 		= ModelFactory.create(model_params.MODEL_PARAMS)
		model.enableInference({'predictedField' : 'binAmplitude'})
		likelihoods = AnomalyLikelihood()
		shifter 	= InferenceShifter()

		"""
		Create vectors to hold data
		"""
		actHistory 	= deque([0.0] * xLimit, maxlen = 60)
		predHistory	= deque([0.0] * xLimit, maxlen = 60)
		anomHistory = deque([0.0] * xLimit, maxlen = 60)
		likeHistory	= deque([0.0] * xLimit, maxlen = 60)

		"""
		4 Lines to plot the Actual input, Predicted input, Anomaly and Likelihood
		"""
		actline, 	= plt.plot(range(xLimit), actHistory)
		predline, 	= plt.plot(range(xLimit), predHistory)
		anomline,	= plt.plot(range(xLimit), anomHistory)
		likeline,	= plt.plot(range(xLimit), likeHistory)	

		"""
		Start the execution of audio stream
		"""
		audio = AudioStream()

		while True:

			"""
			The input is the second bin ([1]), which represents the amplitude of 
			frequencies ranging from (n*sr / bufferSize) to ((n+1)*sr / bufferSize) 
			where n is the bin number selected as input. 
			In this case n = 1 and the range is from 10.67Hz to 21.53Hz
			"""
			inputLevel	= audio.audioFFT[1]

			# Clip input
			maxLevel = model_params.MODEL_PARAMS['modelParams']['sensorParams']['encoders']['binAmplitude']['maxval'] 
			if inputLevel >  maxLevel:
				inputLevel = maxLevel

			# Run the input through the model and shift the resulting prediction.
			modelInput 	= {'binAmplitude' : inputLevel}
			result 		= shifter.shift(model.run(modelInput))

			# Get inference, anomaly and likelihood from the model
			inference 	= result.inferences['multiStepBestPredictions'][5]
			anomaly 	= result.inferences['anomalyScore']
			likelihood 	= likelihoods.anomalyProbability(inputLevel, anomaly)

			# Add values to the end of corresponding vector to plot them
			# Scale anomaly and likelihood to be visible in the plot
			actHistory .append(result.rawInput['binAmplitude'])
			predHistory.append(inference)
			anomHistory.append(anomaly * yLimit/2)
			likeHistory.append(likelihood * yLimit/2)

			# Update plot and draw
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
		 rate: The sampling rate in Hz of the audio interface being used.
		 bufferSize: The size of the array to which we will save audio segments (2^12 = 4096 is very good)
		 bitResolution: Bit depth of every sample
		"""
		rate			= 44100
		self.bufferSize = 2**12
		bitResolution	= 16

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
		non blocking mode. This means the callback executes whenever it needs 
		new audio data (to play) and/or when there is new (recorded) audio data available. 
		Note that PyAudio calls the callback function in a separate thread. 
		"""
		p = pyaudio.PyAudio()

		def callback(in_data, frame_count, time_info, status):

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

		# Wait for the FFT vector to be created in the first callback execution
		while 1:
			try:				
				self.audioFFT
			except AttributeError:
				pass
			else:
				print "Audiostream started"
				break

		"""
		Print out the audio streamd details
		"""
		print "Sampling rate (Hz):\t" + str(rate)
		print "Bit Depth:\t\t"  + str(bitResolution)
		print "Buffersize:\t\t" + str(self.bufferSize)


	def fft(self, audio):
		"""
		Fast Fourier Transform - 
		Output: the transform of the audio input to frequency domain. 
		Contains the amplitude of each frequency in the audio signal
		frequencies are marked by its position in 'output':
		frequency = index * rate / buffesize
		output.size = bufferSize/2 
		Use only first half of vector since the second is repeated due to 
		symmetry.
		Great info here: http://stackoverflow.com/questions/4364823/how-to-get-frequency-from-fft-result
		"""
		output = numpy.abs(numpy.fft.fft(audio))
		return output [0:int(self.bufferSize/2)]


audiostream = AudioPrediction()

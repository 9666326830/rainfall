Source: [UCSD-E4E/passive-acoustic-biodiversity](https://github.com/UCSD-E4E/passive-acoustic-biodiversity/blob/2cb891707c0d3cd074805830832fcad3985c099c/rain_cal.py#L10-L102)

import samplerate
import resampy

"""
Detect Rain
Adopted from Automatic Identification of Rainfall in Acoustic Recordings by 
Carol Bedoya et al.
"""

"""
Calculate thresholds for rain detection in audio files
:param data_path: path to known rain recordings

:return: min threshold for mean PSD, max threshold for mean PSD, 
		 min threshold for SNR, max threshold for snr
"""
def train(data_path):
	sample_rate = None
	recording = []
	psd_arr = []
	snr_arr = []
	for file in glob.glob(data_path + '*.wav'):
		# load wav data
		try:
			rate, data = wavfile.read(file)
		except Exception as e:
			print('(failed) ' + file)
			print('\t' + str(e))
			continue

		recording = np.asarray(data)
		sample_rate = rate
		# import pdb; pdb.set_trace()
		length = recording.shape[0] / sample_rate
		# print(file)
		# print('sample rate = %d' % sample_rate)
		# print('length = %.1fs' % length)

		# import pdb;pdb.set_trace()
		# Stereo to mono
		if recording.ndim == 2:
			recording = recording.sum(axis=1) / 2

		# Downsample to 44.1 kHz
		if sample_rate != 44100:
			recording = signal.decimate(recording, int(sample_rate/44100))
			sample_rate = 44100

		# STEP 1: Estimate PSD vector from signal vector
		f, p = signal.welch(recording, fs=sample_rate, window='hamming', 
							nperseg=512, detrend=False)
		p = np.log10(p)

		# STEP 2: Extract vector a (freq band where rain lies) from PSD vector
		# min and max freq of rainfall freq band
		# divide by sample_rate to normalize from 0 to 1
		rain_min = (2.0 * 600) / sample_rate
		rain_max = (2.0 * 1200) / sample_rate

		limite_inf = int(round(p.__len__() * rain_min))
		limite_sup = int(round(p.__len__() * rain_max))

		# section of interest of the power spectral density
		a = p[limite_inf:limite_sup]
		# print(limite_inf)
		# print(limite_sup)
		# print(a)

		# STEP 3: Compute c (SNR of the PSD in rain freq band)
		# upper part of algorithm 2.1
		mean_a = np.mean(a)
		# lower part of algorithm 2.1
		std_a = np.std(a)

		# snr
		c = mean_a / std_a

		psd_arr.append(mean_a)
		snr_arr.append(c)

	return min(psd_arr), max(psd_arr), min(snr_arr), max(snr_arr


"""
Classify clips as rain or non-rain
:param data_path: path to recordings
:param min_psd: minimum threshold for the mean value of the PSD
:param max_psd: maximum threshold for the mean value of the PSD
:param min_snr: Minimum threshold for the signal to noise ratio
:param max_snr: Maximum threshold for the signal to noise ratio

:return: indicator for rainy clips, metric of intensity of rainfall
"""
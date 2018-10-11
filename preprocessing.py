"""
Derived from lab and https://github.com/helderm/listnr/blob/master/listnr/timit.py
"""

import numpy as np
import os
from pysndfile import sndio

# class2pho = {
#     'aa': {'idx': 0, 'pho': ['aa', 'ao']},
#     'ah': {'idx': 1, 'pho': ['ah', 'ax', 'ax-h']},
#     'er': { 'idx': 2, 'pho': ['er', 'axr']},
#     'hh': {'idx': 3, 'pho': ['hh', 'hv']},
#     'ih': {'idx': 4, 'pho': ['ih', 'ix']},
#     'l': {'idx': 5, 'pho': ['l', 'el']},
#     'm': {'idx': 6, 'pho': ['m', 'em']},
#     'n': {'idx': 7, 'pho': ['n', 'en', 'nx']},
#     'ng': {'idx': 8, 'pho': ['ng', 'eng']},
#     'sh': {'idx': 9, 'pho': ['sh', 'zh']},
#     'uw': {'idx': 10, 'pho': ['uw', 'ux']},
#     'sil': {'idx': 11, 'pho': ['pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl',
#                                'h#', 'pau', 'epi']},
# }

class2pho = {
    'aa': {'idx': 0, 'pho': ['aa']}
}


def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This can be avoided by
    specifying the dtype argument in sndio.read(). However, when I imported'
    the data in lab 1 and 2, computed features and trained the HMM models,
    I used the default behaviour in sndio.read() and rescaled the samples
    in the int16 range instead. In order to compute features that are
    compatible with the models, we have to follow the same procedure again.
    This will be simplified in future years.
    """
    sndobj = sndio.read(filename)
    samplingrate = sndobj[1]
    samples = np.array(sndobj[0]) * np.iinfo(np.int16).max
    return samples, samplingrate


def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512,
         nceps=13, samplingrate=16000, liftercoeff=22, return_mspec=False):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    if return_mspec:
        return lifter(ceps, liftercoeff), mspec
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    to_return = []
    i = 0

    while (i + winlen) < len(samples):
        to_return.append(np.array(samples[i:i + winlen]))
        i += winshift
    return np.array(to_return)


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.
    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)
    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # TODO gain better understanding of why we use these coefficients.
    from scipy.signal import lfilter
    return lfilter([1, -p], 1, input)


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    from scipy.signal import hamming
    return input * hamming(input.shape[1], sym=False)


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    from scipy.fftpack import fft
    freq = fft(input, nfft)
    return np.abs(freq) ** 2


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    fbank = trfbank(samplingrate, input.shape[1])
    return np.log(input @ fbank.T)


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    from scipy.fftpack.realtransforms import dct
    return dct(input, norm="ortho")[:, :nceps]  # use first nceps coefficients.


def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.

       mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
       lifter: lifering coefficient

    Returns:
       NxM array with lifeterd coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter / 2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    return np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes, nceps)))


def trfbank(fs, nfft, lowfreq=133.33, linsc=200 / 3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.

    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters

    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # ------------------------
    # Compute the filter bank
    # ------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt + 2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt - 1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i + 1]
        hi = freqs[i + 2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank


def Get_phonemes(filename):
    """
    Get the phonemes info from TIMIT
    :param filename: phoneme file
    :return: phoneme info
    """
    global class2pho

    phonemes = []
    with open(filename) as f:
        for line in f:
            parts = line.rstrip().split(' ')
            phoneme = parts[2]
            if phoneme == 'q':
                continue

            # search for the phoneme in th
            found = False
            if phoneme not in class2pho:
                for phm, val in class2pho.items():
                    if phoneme in val['pho']:
                        found = True
                        break
                if found:
                    phoneme = phm
                else:
                    idx = max(class2pho.values(), key=lambda x: x['idx'])['idx']
                    # idx = max(class2pho.iteritems(), key=operator.itemgetter(1))[1]
                    class2pho[phoneme] = {'idx': idx + 1, 'pho': [phoneme]}

            phd = {'start': int(parts[0]),
                   'end': int(parts[1]),
                   'phoneme': phoneme}

            phonemes.append(phd)

    return phonemes


def Get_words(filename):
    """
    Get the words info from TIMIT
    :param sfile: word file
    :return: word info
    """
    words = []
    with open(filename) as f:
        for line in f:
            parts = line.rstrip().split(' ')
            ph = {'start': int(parts[0]),
                  'end': int(parts[1]),
                  'sentence': parts[2:]}

            words.append(ph)

    return words


def Make_target(sample, phonemes, winlen=400, winshift=200):
    # assigne a phoneme to each window frame base on the number of samples in common
    start_win = 0
    index_phoneme = 0
    max_phoneme = len(phonemes)
    target = []
    count = 0
    while (start_win + winlen) < len(sample):
        while phonemes[index_phoneme]['end'] <= start_win:
            index_phoneme += 1
            if index_phoneme % max_phoneme == 0:
                index_phoneme -= 1
                break
        max_phon = 0
        if (index_phoneme + max_phon + 1) < max_phoneme:
            while phonemes[index_phoneme + max_phon + 1]['start'] <= start_win + winlen:
                max_phon += 1
                if (index_phoneme + max_phon + 1) % max_phoneme == 0:
                    break
        selected_phon = index_phoneme
        match_phon = min(0, start_win - phonemes[index_phoneme]['start']) + min(0, phonemes[index_phoneme][
            'end'] - start_win + winlen)
        for possible_phon_index in range(max_phon):
            curr_match_phon = min(0, start_win - phonemes[possible_phon_index]['start']) + min(0, phonemes[
                possible_phon_index]['end'] - start_win + winlen)
            if curr_match_phon > match_phon:
                selected_phon = possible_phon_index
                match_phon = curr_match_phon

        target.append(phonemes[selected_phon]['phoneme'])
        start_win += winshift
        count += 1

    return target

# def Make_target(sample, phonemes, winlen=400):
# 	#assigne a phoneme to each window frame base on the number of samples in common

# 	phone_timestep  = np.array([0])
# 	for phone in phonemes:
# 		phone_timestep = np.append(phone_timestep, phone['end'])
# 	win_timestep = np.array([i*winlen for i in range(len(sample)//winlen+1)])

# 	np.argmax(np.min(0, phone_timestep[1:] - win_timestep[:-1]))

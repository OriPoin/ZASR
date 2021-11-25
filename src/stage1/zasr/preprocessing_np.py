import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
from numpy import fft, int16
from scipy.fftpack import fft
from . import params

# Dataset
# def load_dataset()

# Signal

SAMPLE_RATE = params.HP_SAMPLE_RATE.domain.values[0]
PRE_EMPHASIS = params.HP_PRE_EMPHASIS.domain.values[0]
FRAME_LENGTH = params.HP_FRAME_LENGTH.domain.values[0]
FRAME_STEP = params.HP_FRAME_STEP.domain.values[0]
MEL_BINS = params.HP_MEL_BINS.domain.values[0]
FRAME_LENGTH_T = int(SAMPLE_RATE * FRAME_LENGTH)
FRAME_STEP_T = int(SAMPLE_RATE * FRAME_STEP)


def load_audio(file_name):
    r"""
    load audio from file with certain sample rate
    only use the first channel
    """
    sr, audio = wavfile.read(file_name)
    # at least 2d array
    audio = np.atleast_2d(audio)
    if sr != SAMPLE_RATE:
        audio = signal.resample(audio, SAMPLE_RATE)
    audio = audio[0, :]
    return audio.flatten()


def pre_emphasis(audio):
    temp = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])
    return temp


def frame_split(audio):
    frame_num = int(-(-(audio.size - FRAME_LENGTH_T)//FRAME_STEP_T) + 1)
    sample_num = (frame_num - 1) * FRAME_STEP_T + FRAME_LENGTH_T
    head_padding = int((sample_num-audio.size)//2)
    end_padding = int((sample_num-audio.size)-(sample_num-audio.size)//2)
    padded_audio = np.pad(audio, (head_padding, end_padding))
    index = np.tile(np.arange(0, FRAME_LENGTH_T), (frame_num, 1)) + np.tile(
        np.arange(0, (frame_num) * FRAME_STEP_T, FRAME_STEP_T), (FRAME_LENGTH_T, 1)).T
    return padded_audio[index]


def hamming_window_func(frames):
    return frames * signal.windows.hamming(FRAME_LENGTH_T)


def stft_mag(frames):
    frame_fft = fft(frames)
    frame_power = np.abs(frame_fft)** 2
    frame_power = frame_power/FRAME_LENGTH_T
    mag_specs = frame_power[:, 0:FRAME_LENGTH_T//2]
    return mag_specs


def mel_filter(mag_specs):
    low_freq_mel = 0
    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, MEL_BINS + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((FRAME_LENGTH_T + 1)
                   * hz_points / SAMPLE_RATE)
    fbank = np.zeros((MEL_BINS, int(SAMPLE_RATE * FRAME_LENGTH/2)))
    for m in range(1, MEL_BINS+1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    mel_specs = np.dot(mag_specs, fbank.T)
    return mel_specs


def log_fbanks(mel_specs):
    filter_banks = 10 * np.log10(mel_specs)  # dB
    return filter_banks


def audio_preprocessing(file_name):
    r"""
    input: wav files
    output: logfbanks features
    """
    audio = load_audio(file_name)
    pre_emp_audio = pre_emphasis(audio)
    frames = frame_split(pre_emp_audio)
    windowed_frames = hamming_window_func(frames)
    mag_specs = stft_mag(windowed_frames)
    mel_specs = mel_filter(mag_specs)
    filter_banks = log_fbanks(mel_specs)
    return filter_banks


def txt_preprocessing(file_name):
    trans_file = open(file_name)
    trans_script = trans_file.read()
    trans_chars = list(trans_script)
    if trans_chars[-1] == '\n':
        trans_chars = trans_chars[0:-1]
    c_bit_list = list()
    for c in trans_chars:
        c_bits = list('{:b}'.format(ord(c)).zfill(32))
        c_bits = np.array(list(map(float, c_bits)))
        c_bits = 2 * c_bits - 1
        c_bit_list.append(c_bits)
    trans_bit_array = np.array(c_bit_list)
    return trans_bit_array

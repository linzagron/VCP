from datetime import datetime

import wavio
from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG

from os.path import basename, splitext
import sys
import time

import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
import librosa
import librosa.display
import IPython
from IPython.display import Audio

from os import listdir, path, makedirs, mkdir
from os.path import join, expanduser

DATA_ROOT = join(expanduser("~"), "Documents", "Hit", "GeneralResources", "VCProjectDate", "Training")
print(DATA_ROOT)
print(listdir(DATA_ROOT))


fs = 16000
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 24
frame_period = 5
hop_length = int(fs * (frame_period * 0.001))
max_files = 530  # number of utterances to be used.
test_size = 0.03
use_delta = True

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]


class MyFileDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(MyFileDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        paths = super(
            MyFileDataSource, self).collect_files()
        paths_train, paths_test = train_test_split(
            paths, test_size=test_size, random_state=1234)

        # keep paths for later testing
        self.test_paths = paths_test

        return paths_train

    def collect_features(self, path):
        fs, x = wavfile.read(path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = trim_zeros_frames(spectrogram)
        mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
        return mc


bdl_source = MyFileDataSource(data_root=DATA_ROOT,
                              speakers=["bdl"], max_files=max_files)
slt_source = MyFileDataSource(data_root=DATA_ROOT,
                              speakers=["slt"], max_files=max_files)

X = PaddedFileSourceDataset(bdl_source, 1200).asarray()
Y = PaddedFileSourceDataset(slt_source, 1200).asarray()
print(X.shape)
print(Y.shape)

# # Plotting util
# def plot_parallel(x,y):
#     figure(figsize=(16,7))
#     subplot(2,1,1)
#     librosa.display.specshow(trim_zeros_frames(x).T, sr=fs, hop_length=hop_length, x_axis="time")
#     colorbar()
#     subplot(2,1,2)
#     librosa.display.specshow(trim_zeros_frames(y).T, sr=fs, hop_length=hop_length, x_axis="time")
#     colorbar()
#
# idx = 22 # any
# plot_parallel(X[idx],Y[idx])

# Alignment
X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))

# plot_parallel(X_aligned[idx],Y_aligned[idx])

# Drop 1st (power) dimension
X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]

# Append delta features
static_dim = X_aligned.shape[-1]
if use_delta:
    X_aligned = apply_each2d_trim(delta_features, X_aligned, windows)
    Y_aligned = apply_each2d_trim(delta_features, Y_aligned, windows)

# plot_parallel(X_aligned[idx],Y_aligned[idx])

# Finally, we get joint feature matrix
XY = np.concatenate((X_aligned, Y_aligned), axis=-1).reshape(-1, X_aligned.shape[-1] * 2)
print(XY.shape)

XY = remove_zeros_frames(XY)
print(XY.shape)

# Model
gmm = GaussianMixture(
    n_components=64, covariance_type="full", max_iter=100, verbose=1)

gmm.fit(XY)


# Visualize model
# Means
# for k in range(3):
#     plot(gmm.means_[k], linewidth=1.5, label="Mean of mixture {}".format(k+1))
# legend(prop={"size": 16})

# Covariances
# imshow(gmm.covariances_[0], origin="bottom left")
# colorbar()

# for k in range(3):
#     plot(np.diag(gmm.covariances_[k]), linewidth=1.5,
#          label="Diagonal part of covariance matrix, mixture {}".format(k))
# legend(prop={"size": 16})

# Test
def test_one_utt(src_path, tgt_path, disable_mlpg=False, diffvc=True):
    # GMM-based parameter generation is provided by the library in `baseline` module
    if disable_mlpg:
        # Force disable MLPG
        paramgen = MLPG(gmm, windows=[(0, 0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=windows, diff=diffvc)

    fs, x = wavfile.read(src_path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    if use_delta:
        mc = delta_features(mc, windows)
    mc = paramgen.transform(mc)
    if disable_mlpg and mc.shape[-1] != static_dim:
        mc = mc[:, :static_dim]
    assert mc.shape[-1] == static_dim
    mc = np.hstack((c0[:, None], mc))
    if diffvc:
        mc[:, 0] = 0  # remove power coefficients
        engine = Synthesizer(MLSADF(order=order, alpha=alpha), hopsize=hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
        waveform = engine.synthesis(x, b)
    else:
        spectrogram = pysptk.mc2sp(
            mc.astype(np.float64), alpha=alpha, fftlen=fftlen)
        waveform = pyworld.synthesize(
            f0, spectrogram, aperiodicity, fs, frame_period)

    return waveform


# -----
# tgt_path = "arctic_a0001.wav"
# print(DATA_ROOT + "\\" + "cmu_us_bdl_arctic\\wav\\" + tgt_path)
# fs, x = wavfile.read(DATA_ROOT + "\\" + "cmu_us_bdl_arctic\\wav\\" + tgt_path)
# TEST_ROOT = join(expanduser("~"), "Documents", "Hit", "GeneralResources", "VCProjectDate", "Testing")
# now = datetime.now().strftime("%d-%m-%Y%H-%M-%S")
# if not path.exists(TEST_ROOT):
#     makedirs(TEST_ROOT)
# test_dir = TEST_ROOT + '\\' + now
# mkdir(test_dir)
# wavfile.write(join(test_dir, "wo_MLPG" + tgt_path), rate=fs, data=x)
# -----


TEST_ROOT = join(expanduser("~"), "Documents", "Hit", "GeneralResources", "VCProjectDate", "Testing")
now = datetime.now().strftime("%d-%m-%Y%H-%M-%S")
if not path.exists(TEST_ROOT):
    makedirs(TEST_ROOT)
test_dir = TEST_ROOT + '\\' + now
mkdir(test_dir)

# Listen results
for i, (src_path, tgt_path) in enumerate(zip(bdl_source.test_paths, slt_source.test_paths)):
    print("{}-th sample".format(i + 1))
    wo_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=True)
    w_MLPG = test_one_utt(src_path, tgt_path, disable_mlpg=False)
    _, src = wavfile.read(src_path)
    _, tgt = wavfile.read(tgt_path)

    print("Source:", basename(src_path))
    IPython.display.display(Audio(src, rate=fs))
    # wavfile.write(join(test_dir, ("src" + str(i) + ".wav")), rate=fs, data=src)
    wavio.write(join(test_dir, ("src" + str(i) + ".wav")), src, fs, sampwidth=1)

    print("Target:", basename(tgt_path))
    IPython.display.display(Audio(tgt, rate=fs))
    # wavfile.write(join(test_dir, ("tgt" + str(i) + ".wav")), rate=fs, data=tgt)
    wavio.write(join(test_dir, ("tgt" + str(i) + ".wav")), tgt, fs, sampwidth=1)

    print("w/o MLPG")
    IPython.display.display(Audio(wo_MLPG, rate=fs))
    # wavfile.write(join(test_dir, ("wo_MLPG" + str(i) + ".wav")), rate=fs, data=wo_MLPG)
    wavio.write(join(test_dir, ("wo_MLPG" + str(i) + ".wav")), wo_MLPG, fs, sampwidth=1)

    print("w/ MLPG")
    IPython.display.display(Audio(w_MLPG, rate=fs))
    # wavfile.write(join(test_dir, ("w_MLPG" + str(i) + ".wav")), rate=fs, data=w_MLPG)
    wavio.write(join(test_dir, ("w_MLPG" + str(i) + ".wav")), w_MLPG, fs, sampwidth=1)


from scipy.io import wavfile
from os.path import join, expanduser
import IPython
from IPython.display import Audio
import wavio

TEST_ROOT = join(expanduser("~"), "Documents", "Hit", "GeneralResources", "VCProjectDate", "Testing", "14-01-202113-04-43")

fs, a = wavfile.read(TEST_ROOT + "\\w_MLPG0.wav")
wavio.write(join(TEST_ROOT, "w_MLPG0_.wav"), a, fs, sampwidth=1)

fs, a = wavfile.read(TEST_ROOT + "\\wo_MLPG0.wav")
wavio.write(join(TEST_ROOT, "wo_MLPG0_.wav"), a, fs, sampwidth=1)

fs, a = wavfile.read(TEST_ROOT + "\\w_MLPG1.wav")
wavio.write(join(TEST_ROOT, "w_MLPG1_.wav"), a, fs, sampwidth=1)

fs, a = wavfile.read(TEST_ROOT + "\\wo_MLPG1.wav")
wavio.write(join(TEST_ROOT, "wo_MLPG1_.wav"), a, fs, sampwidth=1)

fs, a = wavfile.read(TEST_ROOT + "\\w_MLPG2.wav")
wavio.write(join(TEST_ROOT, "w_MLPG2_.wav"), a, fs, sampwidth=1)

fs, a = wavfile.read(TEST_ROOT + "\\wo_MLPG2.wav")
wavio.write(join(TEST_ROOT, "wo_MLPG2_.wav"), a, fs, sampwidth=1)

# print(type(a))

# audio = Audio(a, rate=fs1)
# IPython.display.display(audio)
# wavfile.write(join(TEST_ROOT, "o1.wav"), rate=fs1, data=audio)

# Write the samples to a file
# wavio.write(join(TEST_ROOT, "o2A.wav"), a, fs1, sampwidth=1)

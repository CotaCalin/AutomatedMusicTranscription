import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def cqt(inputFile, outputFile):
  plt.figure(figsize=(7.5, 3.75))
  y, sr = librosa.load(inputFile)
  C = librosa.cqt(y, sr=sr)
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                            sr=sr)
  plt.axis('off')
  plt.savefig(outputFile, bbox_inches="tight")
  plt.close('all')

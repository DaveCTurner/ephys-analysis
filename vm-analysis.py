#!/usr/bin/python
# coding: utf-8

from neo.io import AxonIO
from numpy import mean, std, arange, convolve, ones, amin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from math import floor, sqrt
import quantities as pq
from glob import glob
import datetime
import scipy
import cmath
from scipy.signal import butter, lfilter

# Define colour map: 'winter' is kinda green to kinda blue.
cmap = cm.get_cmap('winter')

datasets = glob('data\\vm\\*')

all_signals = {}

xRange = arange(0, 60, 1.0/20000.0)

for dataset in datasets:
  print('Loading dataset "' + dataset + '"')
  filenames = glob(dataset + '\\*.abf')
  all_signals[dataset] = {}
  
  fig = plt.figure(figsize=(20,10),dpi=80)
  plt.title(dataset)
  
  signalCount      = 0
    
  for filename in filenames:
    print('Loading file "' + filename + '"')
    reader = AxonIO(filename=filename)
    blocks = reader.read()
    assert len(blocks) == 1
    assert len(blocks[0].segments) == 1
    assert len(blocks[0].segments[0].analogsignals) == 1
    
    sig = blocks[0].segments[0].analogsignals[0]
    
    assert(sig.sampling_rate == pq.Quantity(20000, 'Hz'))
    assert(len(sig) == 60*20000)
    
    all_signals[dataset][filename] = sig
    line = plt.plot(xRange, sig, linewidth=0.5, alpha=0.50)
    plt.setp(line, color='#000000')
    
    if (signalCount == 0):
      sumSignal = sig
      sumSignalSquared = sig * sig
    else:
      sumSignal += sig
      sumSignalSquared += sig * sig
    signalCount += 1
   
  if signalCount == 0:
    continue

  mean = sumSignal / signalCount
  var  = sumSignalSquared / signalCount - mean * mean
  std  = pq.Quantity([sqrt(v.item()) for v in var], 'mV')
  lb   = mean - std
  ub   = mean + std
    
  line = plt.plot(xRange, mean, lineWidth=1.0, alpha=1.0)
  plt.setp(line, color='#ff0000')

  line = plt.fill_between(xRange, lb, ub, alpha = 0.3)
  plt.setp(line, color='#00ff00')
  
  plt.grid()
  plt.savefig(dataset + ".png")
 
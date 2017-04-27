#!/usr/bin/python
# coding: utf-8

from neo.io import AxonIO
from numpy import mean, std, arange, convolve, ones, amin, argmin
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from math import floor, sqrt
import quantities as pq
from glob import glob
import datetime
import argparse
import csv
from os.path import basename
import ephysutils
import os

# Handle command-line arguments
parser = argparse.ArgumentParser(description='Vm analysis')
parser.add_argument('path')
args = parser.parse_args()

# Find trace files
traceFilesByExperiment = ephysutils.findTraceFiles(searchRoot = args.path, \
     cellDetailsByCell = ephysutils.loadCellDetails('cell-details.txt'))

# Open a results file with the date in the filename
resultsDirectory = ephysutils.makeResultsDirectory()

pertracefilename = os.path.join(resultsDirectory, 'Vm-results.txt')
pertracefile = open(pertracefilename, 'w')
pertracefile.write('\t'.join(['Path'
                             ,'Filename'
                             ,'Experiment'
                             ,'Cell line'
                             ,'Cell source'
                             ,'Freshness'
                             ,'Cell_identity'
                             ,'Mean Vm (mV)'
                             ]) + '\n')

xRange = arange(0, 60, 1.0/20000.0)
offset = pq.Quantity(-3.8, 'mV')

for experiment in traceFilesByExperiment:
  traceFilesByCondition = traceFilesByExperiment[experiment].get('Vm', None)
  if traceFilesByCondition == None:
    continue

  for condition in traceFilesByCondition:
    os.makedirs(os.path.join(resultsDirectory, experiment, condition))

    signalCount = 0
    conditionData = traceFilesByCondition[condition]
    conditionFiles = conditionData['files']

    for fileWithDetails in conditionFiles:
      cellDetails = fileWithDetails['details']
      
      if cellDetails['classification'] == 'DISCARD':
        continue

      sampleName = os.path.join(experiment, condition, cellDetails['filename'], cellDetails['classification'] or '')

      print ("Analysing", sampleName)

      reader = AxonIO(filename=fileWithDetails['filename'])
      blocks = reader.read()
      assert len(blocks) == 1
      assert len(blocks[0].segments) == 1
      assert len(blocks[0].segments[0].analogsignals) == 1

      signal = blocks[0].segments[0].analogsignals[0] + offset

      assert(signal.sampling_rate == pq.Quantity(20000, 'Hz'))
      assert(len(signal) == 60*20000)

      cellDetails['vm_trace'] = signal

      if (signalCount == 0):
        sumSignal = signal
        sumSignalSquared = signal * signal
      else:
        sumSignal += signal
        sumSignalSquared += signal * signal

      signalCount += 1

    if signalCount == 0:
      continue

    mean = sumSignal / signalCount
    var  = sumSignalSquared / signalCount - mean * mean
    std  = pq.Quantity([sqrt(v.item()) for v in var], 'mV')
    ster = std / sqrt(signalCount)
    lb   = mean - ster
    ub   = mean + ster

    conditionData['mean_voltage']           = mean
    conditionData['voltage_standard_error'] = ster

    fig = plt.figure(figsize=(20,10),dpi=80)
    plt.title(experiment + ' ' + condition)
    plt.xlabel('Time (s)')
    plt.ylabel('Vm (mV)')
    axes = plt.gca()
    axes.set_ylim([-180,15])

    for fileWithDetails in conditionFiles:
      cellDetails = fileWithDetails['details']
      if cellDetails['classification'] == 'DISCARD':
        continue

      line = plt.plot(xRange, cellDetails['vm_trace'], linewidth=0.5, alpha=0.50, zorder=0)
      plt.setp(line, color='#000000')

    line = plt.fill_between(xRange, mean - ster, mean + ster, alpha = 0.3, zorder=1)
    plt.setp(line, color='#00ff00')

    line = plt.plot(xRange, mean, lineWidth=1.0, alpha=1.0, zorder=2)
    plt.setp(line, color='#ff0000')

    plt.grid()
    plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'Vm-all.png'))
    plt.close()

    for fileWithDetails in conditionFiles:
      details = fileWithDetails['details']
      filename = details['filename']
      if details['classification'] == 'DISCARD':
        continue

      fig = plt.figure(figsize=(20,10),dpi=80)
      plt.title(experiment + ' ' + condition + ' ' + filename)
      plt.xlabel('Time (s)')
      plt.ylabel('Vm (mV)')
      axes = plt.gca()
      axes.set_ylim([-180,15])

      signal = details['vm_trace']

      line = plt.plot(xRange, signal, linewidth=1.0, alpha=1.0, zorder=1)
      plt.setp(line, color='#000000')

      meanVm = numpy.mean(signal)
      meanVm.units = 'mV'
      plt.axhline(meanVm, color='#ff0000', alpha=0.5, zorder=0)

      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'Vm-' + filename + '.png'))
      plt.close()

      pertracefile.write('\t'.join( \
        [details['path']
        ,details['filename']
        ,details['experiment']
        ,details['cell_line']
        ,details['cell_source']
        ,details['freshness']
        ,details['cell_identity']
        ,str(meanVm.item())]) + '\n')

pertracefile.close()
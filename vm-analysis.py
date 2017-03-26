#!/usr/bin/python
# coding: utf-8

from neo.io import AxonIO
from numpy import mean, std, arange, convolve, ones, amin, argmin
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

# Handle command-line arguments
parser = argparse.ArgumentParser(description='IV analysis')
parser.add_argument('path')
args = parser.parse_args()

# Load cell-details.txt
cellDetailsByCell = {}
with open('cell-details.txt') as cellDetailsFile:
  cellDetailsReader = csv.DictReader(cellDetailsFile, delimiter='\t')
  for cellDetailsRow in cellDetailsReader:
    if (cellDetailsRow['path'] != ''):
      cellDetailsByCell[cellDetailsRow['filename']] = \
        { 'filename':               cellDetailsRow['filename']                      \
        , 'path':                   cellDetailsRow['path']                          \
        , 'cell_line':              cellDetailsRow['cell_line']                     \
        , 'cell_source':            cellDetailsRow['cell_source']                   \
        , 'protocol':               cellDetailsRow['protocol']                      \
        , 'freshness':              cellDetailsRow['freshness']                     \
        , 'classification':         cellDetailsRow['classification']                \
        , 'date':                   cellDetailsRow['date']                          \
        , 'notes':                  cellDetailsRow['notes']                         \
        }

filenames = glob(args.path + '\\**\\*.abf', recursive=True)

conditions = {}

for filename in filenames:
  cellDetails = cellDetailsByCell.get(basename(filename), None)
  if (cellDetails == None):
    print("No cell details for", filename)
    continue

  if (cellDetails['protocol'] != 'Vm'):
    continue

  if (cellDetails['classification'] == 'DISCARD'):
    continue

  condition = " ".join([cellDetails['cell_source'], cellDetails['cell_line'], cellDetails['freshness']])

  conditionData = conditions.get(condition, None)
  if (conditionData == None):
    conditionData = {'files': []}
    conditions[condition] = conditionData

  conditionData['files'].append({'filename':filename, 'details':cellDetails})

xRange = arange(0, 60, 1.0/20000.0)

for conditionName, conditionData in conditions.items():
  print("Processing", conditionName)

  fig = plt.figure(figsize=(20,10),dpi=80)
  plt.title(conditionName)
  plt.xlabel('Time (s)')
  plt.ylabel('Vm (mV)')
  axes = plt.gca()
  axes.set_ylim([-60,15])

  signalCount = 0

  for file in conditionData['files']:
    print('Loading file "' + file['filename'] + '"')
    reader = AxonIO(filename=file['filename'])
    blocks = reader.read()
    assert len(blocks) == 1
    assert len(blocks[0].segments) == 1
    assert len(blocks[0].segments[0].analogsignals) == 1

    sig = blocks[0].segments[0].analogsignals[0]

    assert(sig.sampling_rate == pq.Quantity(20000, 'Hz'))
    assert(len(sig) == 60*20000)

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
  ster = std / sqrt(signalCount)
  lb   = mean - ster
  ub   = mean + ster

  line = plt.plot(xRange, mean, lineWidth=1.0, alpha=1.0)
  plt.setp(line, color='#ff0000')

  line = plt.fill_between(xRange, lb, ub, alpha = 0.3)
  plt.setp(line, color='#00ff00')

  plt.grid()
  plt.savefig("Vm Results\\Vm " + conditionName + ".png")
  plt.close()

# Open a results file with the date in the filename
rundate = datetime.datetime.utcnow().replace(microsecond=0) \
                          .isoformat('-').replace(':','-')

pertracefilename = 'Vm results\\' + rundate + '-results-per-trace.txt'
print ("Writing per-trace results to", pertracefilename)
pertracefile = open(pertracefilename, 'w')
pertracefile.write('\t'.join(['Path'
                             ,'Filename'
                             ,'Cell line'
                             ,'Cell source'
                             ,'Freshness'
                             ,'Mean Vm (mV)']) + '\n')

for conditionName, conditionData in conditions.items():
  for file in conditionData['files']:
    print('Plotting file "' + file['filename'] + '"')

    fig = plt.figure(figsize=(20,10),dpi=80)
    plt.title(conditionName)
    plt.xlabel('Time (s)')
    plt.ylabel('Vm (mV)')
    axes = plt.gca()
    axes.set_ylim([-60,15])

    reader = AxonIO(filename=file['filename'])
    blocks = reader.read()
    assert len(blocks) == 1
    assert len(blocks[0].segments) == 1
    assert len(blocks[0].segments[0].analogsignals) == 1

    sig = blocks[0].segments[0].analogsignals[0]

    assert(sig.sampling_rate == pq.Quantity(20000, 'Hz'))
    assert(len(sig) == 60*20000)

    line = plt.plot(xRange, sig, linewidth=0.5)
    plt.setp(line, color='#000000')

    details = file['details']

    meanVm = mean(sig)
    meanVm.units = 'mV'

    plt.axhline(meanVm, color='#ff0000', alpha=0.5)

    plt.grid()
    plt.savefig("Vm Results\\Individuals\\Vm " + conditionName + " - " + details['filename'] + ".png")
    plt.close()

    pertracefile.write('\t'.join( \
      [details['path']
      ,details['filename']
      ,details['cell_line']
      ,details['cell_source']
      ,details['freshness']
      ,str(meanVm.item())]) + '\n')


pertracefile.close()
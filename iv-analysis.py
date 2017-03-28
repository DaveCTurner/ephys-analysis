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
import os.path
import ephysutils

# Handle command-line arguments
parser = argparse.ArgumentParser(description='IV analysis')
parser.add_argument('path')
args = parser.parse_args()

# Find trace files
traceFilesByExperiment = ephysutils.findTraceFiles(searchRoot = args.path, \
     cellDetailsByCell = ephysutils.loadCellDetails('cell-details.txt'))

# Define colour map: 'winter' is kinda green to kinda blue.
cmap = cm.get_cmap('winter')

# Define time points for the analysis
tBaselineStart  = 0.245  # Start estimating the baseline from here
tBaselineLength = 0.01   # Estimate the baseline for this long
tEnd            = 0.268  # End the graph
tAnalyseFrom    = 0.2557 # Look for peaks after this time
tAnalyseTo      = 0.263  # Look for peaks before this time

# Open a results file with the date in the filename
resultsDirectory = ephysutils.makeResultsDirectory()

pertracefilename = os.path.join(resultsDirectory, 'results-per-trace.txt')
print ("Writing per-trace results to", pertracefilename)
pertracefile = open(pertracefilename, 'w')
pertracefile.write('\t'.join(['Path'
                             ,'Filename'
                             ,'Cell line'
                             ,'Cell source'
                             ,'Freshness'
                             ,'Segment number'
                             ,'Voltage(mV)'
                             ,'Driving force for Na+ (mV)'
                             ,'Time to peak (s)'
                             ,'I_min(pA)'
                             ,'WCC (pF)'
                             ,'Peak current density(pA/pF)'
                             ,'Classification'
                             ,'Negative noise peak(pA)'
                             ,'Positive noise peak(pA)']) + '\n')

percellfilename = os.path.join(resultsDirectory, 'results-per-cell.txt')
print ("Writing per-cell results to", percellfilename)
percellfile = open(percellfilename, 'w')
percellfile.write('Filename\tBest peak (pA)\tMean RMS noise (pA)\tMean P2P noise\n')

# These traces were done at 50 kHz (see assertion below)
sample_frequency = pq.Quantity(50000.0, 'Hz')
sample_time_sec = (pq.Quantity(1.0, 'Hz') / sample_frequency).item()
segment_count   = 23

def selectTimeRange(signal, signalStartTime, selectionStartTime, selectionEndTime):
  return signal [ (selectionStartTime - signalStartTime) / sample_time_sec
                : (selectionEndTime - signalStartTime) / sample_time_sec
                ]

def voltageFromSegmentIndex(segmentIndex):
  return pq.Quantity(5 * segmentIndex - 85, 'mV')

for experiment in traceFilesByExperiment:
  traceFilesByCondition = traceFilesByExperiment[experiment].get('IV', None)
  if traceFilesByCondition == None:
    continue

  for condition in traceFilesByCondition:
    for fileWithDetails in traceFilesByCondition[condition]:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']

      sampleName = os.path.join(experiment, condition, cellDetails['filename'])

      print ("Analysing", sampleName)

      # Read the file into 'blocks'
      reader = AxonIO(filename=filename)
      blocks = reader.read()

      assert len(blocks) == 1
      assert len(blocks[0].segments) == segment_count

      # Per-cell statistics
      perCellMinPeakSoFar         = pq.Quantity(0, 'pA')
      perCellMinConductanceSoFar  = pq.Quantity(0, 'pA/pF/mV')
      perCellRunningTotalP2PNoise = 0
      perCellRunningTotalRMSNoise = 0

      cellDetails['segments'] = []
      for segmentIndex in range(segment_count):

        thisSegmentData = {}
        cellDetails['segments'].append(thisSegmentData)

        segment = blocks[0].segments[segmentIndex]

        # Use the corrected signal [1] as opposed to the uncorrected [0]
        assert len(segment.analogsignals) == 2
        signal = segment.analogsignals[1]
        assert(signal.units == pq.Quantity(1.0, 'pA'))

        assert(signal.sampling_rate == sample_frequency)
        # assert(len(signal) == 60*50000) ## TODO

        # Estimate the baseline from the quiescent signal
        quiescentSignalWithoutOffset = selectTimeRange(signal, 0, tBaselineStart, tBaselineStart + tBaselineLength)
        baseline                     = mean(quiescentSignalWithoutOffset)
        signal                       = signal - baseline

        # Analyse the noise from the quiescent signal (after applying the offset)
        quiescentSignal = quiescentSignalWithoutOffset - baseline
        thisSegmentData['peakNoiseNeg'] = min(quiescentSignal)
        thisSegmentData['peakNoisePos'] = max(quiescentSignal)
        meanSquareNoise = mean(quiescentSignal**2)
        meanSquareNoise.units = 'pA**2'
        thisSegmentData['rmsNoise'] = sqrt(meanSquareNoise)

        # Only take the signal from tStart to tEnd and take out the estimated baseline
        thisSegmentData['traceToDraw']    = selectTimeRange(signal, 0, tBaselineStart + tBaselineLength, tAnalyseTo)
        toAnalyse                         = selectTimeRange(signal, 0, tAnalyseFrom,                     tAnalyseTo)
        thisSegmentData['traceToAnalyse'] = toAnalyse

        # Find the peak index (number of samples), current and time
        minIndex   = argmin(toAnalyse)
        thisSegmentData['peak_current'] = toAnalyse[minIndex]
        thisSegmentData['time_to_peak'] = minIndex * sample_time_sec + tAnalyseFrom
        thisSegmentData['peak_current_density'] = thisSegmentData['peak_current'] \
                                                / cellDetails['whole_cell_capacitance']

        thisSegmentData['voltage'] = voltageFromSegmentIndex(segmentIndex)
        thisSegmentData['driving_force'] = thisSegmentData['voltage'] - pq.Quantity(85.1, 'mV')
        thisSegmentData['conductance']   = thisSegmentData['peak_current_density'] \
                                         / thisSegmentData['driving_force']

        if (thisSegmentData['peak_current'] < perCellMinPeakSoFar):
          perCellMinPeakSoFar = thisSegmentData['peak_current']

        if (thisSegmentData['conductance'] < perCellMinConductanceSoFar):
          perCellMinConductanceSoFar = thisSegmentData['conductance']

        perCellRunningTotalP2PNoise += thisSegmentData['peakNoisePos'].item() - thisSegmentData['peakNoiseNeg'].item()
        perCellRunningTotalRMSNoise += thisSegmentData['rmsNoise']

      cellDetails['min_peak_current'] = perCellMinPeakSoFar
      cellDetails['min_conductance']  = perCellMinConductanceSoFar
      cellDetails['mean_p2p_noise']   = perCellRunningTotalP2PNoise / segment_count
      cellDetails['mean_rms_noise']   = perCellRunningTotalRMSNoise / segment_count

pertracefile.close()
percellfile.close()

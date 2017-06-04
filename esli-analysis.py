#!/usr/bin/python
# coding: utf-8

from neo.io import AxonIO
from numpy import mean, std, arange, convolve, ones, amin, argmin
import numpy as np
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
parser.add_argument('--path',         required=True)
parser.add_argument('--cell-details', required=True, dest='cellDetails')
parser.add_argument('--results',      required=True)
args = parser.parse_args()

# Find trace files
traceFilesByExperiment = ephysutils.findTraceFiles(searchRoot = args.path, \
     cellDetailsByCell = ephysutils.loadCellDetails(args.cellDetails))

# Define time points for the analysis
tBaselineStart  = 0.245  # Start estimating the baseline from here
tBaselineLength = 0.01   # Estimate the baseline for this long
tEnd            = 0.268  # End the graph
tAnalyseFrom    = 0.2553 # Look for peaks after this time
tAnalyseTo      = 0.263  # Look for peaks before this time

# Open a results file with the date in the filename
resultsDirectory = ephysutils.makeResultsDirectory(args.results)

pertracefilename = os.path.join(resultsDirectory, 'results-per-trace.txt')
print ("Writing per-trace results to", pertracefilename)
pertracefile = open(pertracefilename, 'w')
pertracefile.write('\t'.join(['Path'
                             ,'Filename'
                             ,'Experiment'
                             ,'Cell line'
                             ,'Cell source'
                             ,'Freshness'
                             ,'Segment number'
                             ,'WCC (pF)'
                             ,'Classification'
                             ,'Negative noise peak(pA)'
                             ,'Positive noise peak(pA)'
                             ]) + '\n')

percellfilename = os.path.join(resultsDirectory, 'results-per-cell.txt')
print ("Writing per-cell results to", percellfilename)
percellfile = open(percellfilename, 'w')
percellfile.write('\t'.join(['Path'
                            ,'Filename'
                            ,'Experiment'
                            ,'Cell line'
                            ,'Cell source'
                            ,'Freshness'
                            ,'Cell identity'
                            ,'Total segments'
                            ,'Segments analysed'
                            ,'WCC (pF)'
                            ,'Mean RMS noise (pA)'
                            ,'Mean P2P noise'
                            ,'Classification'
                            ,'Time to peak (s)'
                            ,'Peak current (pA)'
                            ,'Peak current density (pA/pF)'
                            ]) + '\n')

# These traces were done at 50 kHz (see assertion below)
sample_frequency = pq.Quantity(50000.0, 'Hz')
sample_time_sec = (pq.Quantity(1.0, 'Hz') / sample_frequency).item()

def selectTimeRange(signal, signalStartTime, selectionStartTime, selectionEndTime):
  startIndex = round((selectionStartTime - signalStartTime) / sample_time_sec)
  endIndex   = round((selectionEndTime   - signalStartTime) / sample_time_sec)
  return signal[startIndex:endIndex]

def doNotProcess(cellDetails):
  return False

for experiment in traceFilesByExperiment:
  traceFilesByCondition = traceFilesByExperiment[experiment].get('ESL', None)
  if traceFilesByCondition == None:
    continue

  for condition in traceFilesByCondition:
    os.makedirs(os.path.join(resultsDirectory, experiment, condition))
    conditionActivationVoltage = None
    conditionFiles = traceFilesByCondition[condition]['files']

    for fileWithDetails in conditionFiles:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']

      sampleName = os.path.join(experiment, condition, cellDetails['filename'], cellDetails['classification'])

      print ("Analysing", sampleName)

      if conditionActivationVoltage is not None:
          assert(conditionActivationVoltage == cellDetails['activation_voltage'])

      conditionActivationVoltage = cellDetails['activation_voltage']

      # Read the file into 'blocks'
      reader = AxonIO(filename=filename)
      blocks = reader.read()

      assert len(blocks) == 1
      segment_count = len(blocks[0].segments)
      cellDetails['actual_segment_count'] = segment_count
      if segment_count > 18:
        segment_count = 18

      # Per-cell statistics
      perCellRunningTotalP2PNoise = 0
      perCellRunningTotalRMSNoise = 0

      cellDetails['mean_current_trace'] = None
      cellDetails['segments'] = []
      for segmentIndex in range(segment_count):

        thisSegmentData = {'segmentIndex': segmentIndex}
        cellDetails['segments'].append(thisSegmentData)

        segment = blocks[0].segments[segmentIndex]

        # Use the corrected signal [1] as opposed to the uncorrected [0]
        assert len(segment.analogsignals) == 2
        signal = segment.analogsignals[1]
        assert(signal.units == pq.Quantity(1.0, 'pA'))

        assert(signal.sampling_rate == sample_frequency)

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

        if (cellDetails['mean_current_trace'] is None):
          cellDetails['mean_current_trace'] = thisSegmentData['traceToDraw'] / segment_count # copy
        else:
          cellDetails['mean_current_trace'] += thisSegmentData['traceToDraw'] / segment_count

        perCellRunningTotalP2PNoise += thisSegmentData['peakNoisePos'].item() - thisSegmentData['peakNoiseNeg'].item()
        perCellRunningTotalRMSNoise += thisSegmentData['rmsNoise']

      # Find the peak index (number of samples), current and time
      minIndex   = argmin(cellDetails['mean_current_trace'])
      cellDetails['peak_current'] = cellDetails['mean_current_trace'][minIndex]
      cellDetails['time_to_peak'] = minIndex * sample_time_sec + tBaselineStart + tBaselineLength
      cellDetails['peak_current_density'] = cellDetails['peak_current'] \
                                          / cellDetails['whole_cell_capacitance']

      cellDetails['mean_p2p_noise']   = perCellRunningTotalP2PNoise / segment_count
      cellDetails['mean_rms_noise']   = perCellRunningTotalRMSNoise / segment_count

      # Write the per-cell results
      percellfile.write('\t'.join([cellDetails['path']
                                  ,cellDetails['filename']
                                  ,cellDetails['experiment']
                                  ,cellDetails['cell_line']
                                  ,cellDetails['cell_source']
                                  ,cellDetails['freshness']
                                  ,cellDetails['cell_identity']
                                  ,str(cellDetails['actual_segment_count'])
                                  ,str(len(cellDetails['segments']))
                                  ,str(cellDetails['whole_cell_capacitance'].item())
                                  ,str(cellDetails['mean_rms_noise'])
                                  ,str(cellDetails['mean_p2p_noise'])
                                  ,cellDetails['classification']
                                  ,str(cellDetails['time_to_peak'].item())
                                  ,str(cellDetails['peak_current'].item())
                                  ,str(cellDetails['peak_current_density'].item())
                                  ]) + '\n')

      # Write the per-trace results
      for thisSegmentData in cellDetails['segments']:
        pertracefile.write('\t'.join([cellDetails['path']
                                     ,cellDetails['filename']
                                     ,cellDetails['experiment']
                                     ,cellDetails['cell_line']
                                     ,cellDetails['cell_source']
                                     ,cellDetails['freshness']
                                     ,str(thisSegmentData['segmentIndex'])
                                     ,str(cellDetails['whole_cell_capacitance'].item())
                                     ,cellDetails['classification']
                                     ,str(thisSegmentData['peakNoiseNeg'].item())
                                     ,str(thisSegmentData['peakNoisePos'].item())
                                     ]) + '\n')

      # Draw the traces for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Time (s)')
      plt.ylabel('I (pA)')
      axes = plt.gca()
      axes.set_ylim([-670,100])

      for thisSegmentData in cellDetails['segments']:
        thisSegmentColor = '#808080'

        line = plt.plot([tBaselineStart + tBaselineLength + sample_index * sample_time_sec
                            for sample_index in range(len(thisSegmentData['traceToDraw']))],
                        thisSegmentData['traceToDraw'], linewidth=0.5, alpha=0.5)
        plt.setp(line, color=thisSegmentColor)

      line = plt.plot([tBaselineStart + tBaselineLength + sample_index * sample_time_sec
                          for sample_index in range(len(cellDetails['mean_current_trace']))],
                      cellDetails['mean_current_trace'], linewidth=1.0, alpha=1.0)
      plt.setp(line, color='#000000')

      mark = plt.plot(cellDetails['time_to_peak'], cellDetails['peak_current'], '+')
      plt.setp(mark, color='#000000')

      plt.axvspan(tAnalyseFrom, tAnalyseTo, facecolor='#c0c0c0', alpha=0.5)
      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'iv-traces-' + cellDetails['filename'] + ".png"))
      plt.close()

pertracefile.close()
percellfile.close()

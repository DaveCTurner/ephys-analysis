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
tAnalyseFrom    = 0.2557 # Look for peaks after this time
tAnalyseTo      = 0.263  # Look for peaks before this time
tPersistentFrom = 0.300  # Start measuring mean persistent current
tPersistentLength   = 0.015  # Measure mean persistent current for this long

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
                             ,'Cell identity'
                             ,'Segment number'
                             ,'Voltage(mV)'
                             ,'Use peak or mean'
                             ,'Driving force for Na+ (mV)'
                             ,'Time to peak (s)'
                             ,'I_min(pA)'
                             ,'I_mean(pA)'
                             ,'I_selected(pA)'
                             ,'WCC (pF)'
                             ,'Peak current density(pA/pF)'
                             ,'Mean current density(pA/pF)'
                             ,'Selected current density(pA/pF)'
                             ,'Peak conductance(pA/pF/mV)'
                             ,'Max conductance(pA/pF/mV)'
                             ,'Normalised conductance'
                             ,'Classification'
                             ,'Negative noise peak(pA)'
                             ,'Positive noise peak(pA)'
                             ,'Mean persistent current(pA)'
                             ,'Mean persistent current density (pA/pF)'
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
                            ,'WCC (pF)'
                            ,'Best peak (pA)'
                            ,'Mean RMS noise (pA)'
                            ,'Mean P2P noise'
                            ,'Max Conductance (pA/pF/mV)'
                            ,'Classification'
                            ]) + '\n')

# These traces were done at 50 kHz (see assertion below)
sample_frequency = pq.Quantity(50000.0, 'Hz')
sample_time_sec = (pq.Quantity(1.0, 'Hz') / sample_frequency).item()
segment_count   = 23

# Define colour map for traces 1..23 using 'winter' colour scale (green to blue)
colormap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=segment_count), cmap=cm.get_cmap('winter'))

def selectTimeRange(signal, signalStartTime, selectionStartTime, selectionEndTime):
  startIndex = round((selectionStartTime - signalStartTime) / sample_time_sec)
  endIndex   = round((selectionEndTime   - signalStartTime) / sample_time_sec)
  return signal[startIndex:endIndex]

def voltageFromSegmentIndex(segmentIndex):
  return pq.Quantity(5 * segmentIndex - 80, 'mV')

def doNotProcess(cellDetails):
  return cellDetails['classification'] != 'SMALL' \
     and cellDetails['classification'] != 'LARGE'

for experiment in traceFilesByExperiment:
  traceFilesByCondition = traceFilesByExperiment[experiment].get('IV', None)
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
      assert len(blocks[0].segments) == segment_count

      # Per-cell statistics
      perCellMinPeakSoFar         = pq.Quantity(0, 'pA')
      perCellMaxConductanceSoFar  = pq.Quantity(0, 'pA/pF/mV')
      perCellRunningTotalP2PNoise = 0
      perCellRunningTotalRMSNoise = 0

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
        # assert(len(signal) == 60*50000) ## TODO

        # Estimate the baseline from the quiescent signal
        quiescentSignalWithoutOffset = selectTimeRange(signal, 0, tBaselineStart, tBaselineStart + tBaselineLength)
        baseline                     = mean(quiescentSignalWithoutOffset)
        signal                       = signal - baseline

        # Calculate the mean persistent current
        persistentCurrent            = selectTimeRange(signal, 0, tPersistentFrom, tPersistentFrom + tPersistentLength)
        thisSegmentData['mean_persistent_current'] = mean(persistentCurrent)

        # Analyse the noise from the quiescent signal (after applying the offset)
        quiescentSignal = quiescentSignalWithoutOffset - baseline
        thisSegmentData['peakNoiseNeg'] = min(quiescentSignal)
        thisSegmentData['peakNoisePos'] = max(quiescentSignal)
        meanSquareNoise = mean(quiescentSignal**2)
        meanSquareNoise.units = 'pA**2'
        thisSegmentData['rmsNoise'] = sqrt(meanSquareNoise)

        # Only take the signal from tStart to tEnd and take out the estimated baseline
        thisSegmentData['traceToDraw']    = selectTimeRange(signal, 0, tBaselineStart + tBaselineLength, tPersistentFrom + tPersistentLength)
        toAnalyse                         = selectTimeRange(signal, 0, tAnalyseFrom,                     tAnalyseTo)

        # Find the peak index (number of samples), current and time
        minIndex   = argmin(toAnalyse)
        thisSegmentData['peak_current'] = toAnalyse[minIndex]
        thisSegmentData['time_to_peak'] = minIndex * sample_time_sec + tAnalyseFrom
        thisSegmentData['peak_current_density'] = thisSegmentData['peak_current'] \
                                                / cellDetails['whole_cell_capacitance']
        thisSegmentData['persistent_current_density'] = thisSegmentData['mean_persistent_current'] \
                                                / cellDetails['whole_cell_capacitance']


        thisSegmentData['mean_current']         = mean(toAnalyse)
        thisSegmentData['mean_current_density'] = thisSegmentData['mean_current'] \
                                                / cellDetails['whole_cell_capacitance']

        thisSegmentData['voltage'] = voltageFromSegmentIndex(segmentIndex)
        thisSegmentData['driving_force'] = thisSegmentData['voltage'] - pq.Quantity(85.1, 'mV')
        thisSegmentData['conductance']   = thisSegmentData['peak_current_density'] \
                                         / thisSegmentData['driving_force']

        thisSegmentData['peak_or_mean'] = 'mean' if thisSegmentData['voltage'] <= conditionActivationVoltage else 'peak'
        thisSegmentData['selected_current']         = thisSegmentData[thisSegmentData['peak_or_mean'] + '_current']
        thisSegmentData['selected_current_density'] = thisSegmentData[thisSegmentData['peak_or_mean'] + '_current_density']

        if (thisSegmentData['peak_current'] < perCellMinPeakSoFar):
          perCellMinPeakSoFar = thisSegmentData['peak_current']

        if (perCellMaxConductanceSoFar < thisSegmentData['conductance']):
          perCellMaxConductanceSoFar = thisSegmentData['conductance']

        perCellRunningTotalP2PNoise += thisSegmentData['peakNoisePos'].item() - thisSegmentData['peakNoiseNeg'].item()
        perCellRunningTotalRMSNoise += thisSegmentData['rmsNoise']

      cellDetails['min_peak_current'] = perCellMinPeakSoFar
      cellDetails['max_conductance']  = perCellMaxConductanceSoFar
      cellDetails['mean_p2p_noise']   = perCellRunningTotalP2PNoise / segment_count
      cellDetails['mean_rms_noise']   = perCellRunningTotalRMSNoise / segment_count

      for thisSegmentData in cellDetails['segments']:
        thisSegmentData['normalised_conductance'] = thisSegmentData['conductance'] / cellDetails['max_conductance']

      # Write the per-cell results
      percellfile.write('\t'.join([cellDetails['path']
                                  ,cellDetails['filename']
                                  ,cellDetails['experiment']
                                  ,cellDetails['cell_line']
                                  ,cellDetails['cell_source']
                                  ,cellDetails['freshness']
                                  ,cellDetails['cell_identity']
                                  ,str(cellDetails['whole_cell_capacitance'].item())
                                  ,str(cellDetails['min_peak_current'].item())
                                  ,str(cellDetails['mean_rms_noise'])
                                  ,str(cellDetails['mean_p2p_noise'])
                                  ,str(cellDetails['max_conductance'].item())
                                  ,cellDetails['classification']
                                  ]) + '\n')

      # Write the per-trace results
      for thisSegmentData in cellDetails['segments']:
        pertracefile.write('\t'.join([cellDetails['path']
                                     ,cellDetails['filename']
                                     ,cellDetails['experiment']
                                     ,cellDetails['cell_line']
                                     ,cellDetails['cell_source']
                                     ,cellDetails['freshness']
                                     ,cellDetails['cell_identity']
                                     ,str(thisSegmentData['segmentIndex'])
                                     ,str(thisSegmentData['voltage'].item())
                                     ,thisSegmentData['peak_or_mean']
                                     ,str(thisSegmentData['driving_force'].item())
                                     ,str(thisSegmentData['time_to_peak'].item())
                                     ,str(thisSegmentData['peak_current'].item())
                                     ,str(thisSegmentData['mean_current'].item())
                                     ,str(thisSegmentData['selected_current'].item())
                                     ,str(cellDetails['whole_cell_capacitance'].item())
                                     ,str(thisSegmentData['peak_current_density'].item())
                                     ,str(thisSegmentData['mean_current_density'].item())
                                     ,str(thisSegmentData['selected_current_density'].item())
                                     ,str(thisSegmentData['conductance'].item())
                                     ,str(cellDetails['max_conductance'].item())
                                     ,str(thisSegmentData['normalised_conductance'].item())
                                     ,cellDetails['classification']
                                     ,str(thisSegmentData['peakNoiseNeg'].item())
                                     ,str(thisSegmentData['peakNoisePos'].item())
                                     ,str(thisSegmentData['mean_persistent_current'].item())
                                     ,str(thisSegmentData['persistent_current_density'].item())
                                     ]) + '\n')

      # Draw the traces for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Time (s)')
      plt.ylabel('I (pA)')
      axes = plt.gca()
      axes.set_ylim([-670,100])

      for thisSegmentData in cellDetails['segments']:
        thisSegmentColor = colormap.to_rgba(segment_count - thisSegmentData['segmentIndex'])

        line = plt.plot([tBaselineStart + tBaselineLength + sample_index * sample_time_sec
                            for sample_index in range(len(thisSegmentData['traceToDraw']))],
                        thisSegmentData['traceToDraw'], linewidth=0.5, alpha=0.5)
        plt.setp(line, color=thisSegmentColor)

        if thisSegmentData['peak_or_mean'] == 'peak':
          mark = plt.plot(thisSegmentData['time_to_peak'], thisSegmentData['peak_current'], '+')
          plt.setp(mark, color=thisSegmentColor)

      plt.axvspan(tAnalyseFrom, tAnalyseTo, facecolor='#c0c0c0', alpha=0.5)
      plt.axvspan(tPersistentFrom, tPersistentFrom + tPersistentLength, facecolor='#ffc0c0', alpha=0.5)
      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'iv-traces-' + cellDetails['filename'] + ".png"))
      plt.close()

      # Draw the IV curve for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Voltage (mV)')
      plt.ylabel('Current density (pA/pF)')

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['peak_current_density'])

      line = plt.plot(xData, yData, linewidth=0.5, zorder=0)
      plt.setp(line, color='#ff0000')

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['mean_current_density'])

      line = plt.plot(xData, yData, linewidth=0.5, zorder=0)
      plt.setp(line, color='#00ff00')


      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['selected_current_density'])

      line = plt.plot(xData, yData, linewidth=1, zorder=1)
      plt.setp(line, color='#000000')

      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'current-density-' + cellDetails['filename'] + '.png'))
      plt.close()

      # Draw the activation curve for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Voltage (mV)')
      plt.ylabel('Normalised conductance')

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['normalised_conductance'])

      line = plt.plot(xData, yData)
      plt.setp(line, color='#000000')

      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'normalised-conductance-' + cellDetails['filename'] + '.png'))
      plt.close()

    # Draw the IV curves for all cells in this condition, using peak current
    figure = plt.figure(figsize=(20,10), dpi=80)
    plt.title(os.path.join(experiment, condition))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Peak current density (pA/pF)')

    cellCount          = 0
    runningTotal       = np.zeros(segment_count)
    runningSquareTotal = np.zeros(segment_count)

    for fileWithDetails in conditionFiles:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']
      if doNotProcess(cellDetails):
        continue

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['peak_current_density'])

      line = plt.plot(xData, yData, zorder=1)
      plt.setp(line, color='#c0c0c0')

      yData = np.array(yData)
      runningTotal       += yData
      runningSquareTotal += yData * yData
      cellCount          += 1

    if cellCount > 0:
      means     = runningTotal / cellCount
      variances = runningSquareTotal / cellCount - means * means
      stderrs   = [sqrt(var) / sqrt(cellCount)
                  for var in variances]

      plt.errorbar(xData, means, yerr=stderrs, linewidth=0.0, capsize=5.0, color='#000000', capthick=2.0, elinewidth=2.0, marker='o', zorder=2)

    if (conditionActivationVoltage is not None):
      plt.axvspan(-90, conditionActivationVoltage + pq.Quantity(2.5, 'mV'), facecolor='#c0c0c0', alpha=0.5)

    plt.grid()
    plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'peak-current-density-all.png'))
    plt.close()

    # Draw the IV curves for all cells in this condition, using selected current (peak/mean)
    figure = plt.figure(figsize=(20,10), dpi=80)
    plt.title(os.path.join(experiment, condition))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Current density (pA/pF)')

    cellCount          = 0
    runningTotal       = np.zeros(segment_count)
    runningSquareTotal = np.zeros(segment_count)

    for fileWithDetails in conditionFiles:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']
      if doNotProcess(cellDetails):
        continue

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['selected_current_density'])

      line = plt.plot(xData, yData, zorder=1)
      plt.setp(line, color='#c0c0c0')

      yData = np.array(yData)
      runningTotal       += yData
      runningSquareTotal += yData * yData
      cellCount          += 1

    if cellCount > 0:
      means     = runningTotal / cellCount
      variances = runningSquareTotal / cellCount - means * means
      stderrs   = [sqrt(var) / sqrt(cellCount)
                  for var in variances]

      plt.errorbar(xData, means, yerr=stderrs, linewidth=0.0, capsize=5.0, color='#000000', capthick=2.0, elinewidth=2.0, marker='o', zorder=2)

    if (conditionActivationVoltage is not None):
      plt.axvspan(-90, conditionActivationVoltage + pq.Quantity(2.5, 'mV'), facecolor='#c0c0c0', alpha=0.5)

    plt.grid()
    plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'selected-current-density-all.png'))
    plt.close()

    # Draw the activation curves for all cells in this condition
    figure = plt.figure(figsize=(20,10), dpi=80)
    plt.title(os.path.join(experiment, condition))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Normalised conductance')

    cellCount          = 0
    runningTotal       = np.zeros(segment_count)
    runningSquareTotal = np.zeros(segment_count)

    for fileWithDetails in conditionFiles:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']
      if doNotProcess(cellDetails):
        continue

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['normalised_conductance'])

      line = plt.plot(xData, yData, zorder=1)
      plt.setp(line, color='#c0c0c0')

      yData = np.array(yData)
      runningTotal       += yData
      runningSquareTotal += yData * yData
      cellCount          += 1

    if cellCount > 0:
      means     = runningTotal / cellCount
      variances = runningSquareTotal / cellCount - means * means
      stderrs   = [sqrt(var) / sqrt(cellCount)
                  for var in variances]

      plt.errorbar(xData, means, yerr=stderrs, linewidth=0.0, capsize=5.0, color='#000000', capthick=2.0, elinewidth=2.0, marker='o', zorder=2)

    if (conditionActivationVoltage is not None):
      plt.axvspan(-90, conditionActivationVoltage + pq.Quantity(2.5, 'mV'), facecolor='#c0c0c0', alpha=0.5)

    plt.grid()
    plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'normalised-conductance-all.png'))
    plt.close()

pertracefile.close()
percellfile.close()

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
tBaselineStart    = 0.245  # Start estimating the baseline from here
tBaselineLength   = 0.01   # Estimate the baseline for this long
tAnalyseIVFrom    = 0.2595 # Look for IV peaks after this time
tAnalyseIVTo      = 0.266  # Look for IV peaks before this time
tAnalyseSSIFrom   = 0.5096 # Look for SSI peaks after this time
tAnalyseSSITo     = 0.515  # Look for SSI peaks before this time

tPlotIVFrom       = tBaselineStart + tBaselineLength  # Draw IV graph from here
tPlotIVTo         = 0.267  # End the IV graph

tPlotSSIFrom      = 0.508  # Draw SSI graph from here
tPlotSSITo        = 0.518  # End the SSI graph


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
                             ,'When'
                             ,'Segment number'
                             ,'Voltage(mV)'
                             ,'Use peak or mean'
                             ,'Driving force for Na+ (mV)'
                             ,'Time to peak (s)'
                             ,'I_min_IV(pA)'
                             ,'I_mean_IV(pA)'
                             ,'I_selected_IV(pA)'
                             ,'WCC (pF)'
                             ,'Peak current density IV(pA/pF)'
                             ,'Mean current density IV(pA/pF)'
                             ,'Selected current density IV(pA/pF)'
                             ,'Conductance(pA/pF/mV)'
                             ,'Max conductance(pA/pF/mV)'
                             ,'Normalised conductance'
                             ,'Classification'
                             ,'Negative noise peak(pA)'
                             ,'Positive noise peak(pA)'
                             ,'I_min_SSI(pA)'
                             ,'I_mean_SSI(pA)'
                             ,'I_selected_SSI(pA)'
                             ,'Normalised selected current SSI'
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
                            ,'When'
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
segment_count   = 16

# Define colour map for traces 1..16 using 'winter' colour scale (green to blue)
colormap = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=1, vmax=segment_count), cmap=cm.get_cmap('winter'))

def selectTimeRange(signal, signalStartTime, selectionStartTime, selectionEndTime):
  startIndex = round((selectionStartTime - signalStartTime) / sample_time_sec)
  endIndex   = round((selectionEndTime   - signalStartTime) / sample_time_sec)
  return signal[startIndex:endIndex]

def voltageFromSegmentIndex(segmentIndex):
  return pq.Quantity(10 * segmentIndex - 120, 'mV')

def doNotProcess(cellDetails):
  return cellDetails['classification'] != 'SMALL' \
     and cellDetails['classification'] != 'LARGE'

for experiment in traceFilesByExperiment:
  traceFilesByCondition = traceFilesByExperiment[experiment].get('SSI/IV', None)
  if traceFilesByCondition == None:
    continue

  for condition in traceFilesByCondition:
    os.makedirs(os.path.join(resultsDirectory, experiment, condition))
    conditionFiles = traceFilesByCondition[condition]['files']

    for fileWithDetails in conditionFiles:
      filename    = fileWithDetails['filename']
      cellDetails = fileWithDetails['details']

      sampleName = os.path.join(experiment, condition, cellDetails['filename'], cellDetails['classification'])

      print ("Analysing", sampleName)



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

      first_segment_peak_current_SSI = None

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

        # Analyse the noise from the quiescent signal (after applying the offset)
        quiescentSignal = quiescentSignalWithoutOffset - baseline
        thisSegmentData['peakNoiseNeg'] = min(quiescentSignal)
        thisSegmentData['peakNoisePos'] = max(quiescentSignal)
        meanSquareNoise = mean(quiescentSignal**2)
        meanSquareNoise.units = 'pA**2'
        thisSegmentData['rmsNoise'] = sqrt(meanSquareNoise)

        thisSegmentData['traceToDraw_IV']    = selectTimeRange(signal, 0, tPlotIVFrom,  tPlotIVTo)
        thisSegmentData['traceToDraw_SSI']   = selectTimeRange(signal, 0, tPlotSSIFrom, tPlotSSITo)

        # Define region to analyse IV current
        toAnalyseIV = selectTimeRange(signal, 0, tAnalyseIVFrom, tAnalyseIVTo)

        # Find the peak and mean current and current density for the IV trace
        minIndex   = argmin(toAnalyseIV)
        thisSegmentData['peak_current_IV'] = toAnalyseIV[minIndex]
        thisSegmentData['time_to_peak_IV'] = minIndex * sample_time_sec + tAnalyseIVFrom
        thisSegmentData['peak_current_density_IV'] = thisSegmentData['peak_current_IV'] \
                                                   / cellDetails['whole_cell_capacitance']

        thisSegmentData['mean_current_IV']         = mean(toAnalyseIV)
        thisSegmentData['mean_current_density_IV'] = thisSegmentData['mean_current_IV'] \
                                                   / cellDetails['whole_cell_capacitance']

        # Define region to analyse SSI current
        toAnalyseSSI = selectTimeRange(signal, 0, tAnalyseSSIFrom, tAnalyseSSITo)

        # Find the peak and mean current for the SSI trace
        minIndex = argmin(toAnalyseSSI)
        thisSegmentData['peak_current_SSI'] = toAnalyseSSI[minIndex]
        thisSegmentData['mean_current_SSI'] = mean(toAnalyseSSI)
        thisSegmentData['time_to_peak_SSI'] = minIndex * sample_time_sec + tAnalyseSSIFrom


        # Calculate things for the conductance curve
        thisSegmentData['voltage'] = voltageFromSegmentIndex(segmentIndex)
        thisSegmentData['driving_force'] = thisSegmentData['voltage'] - pq.Quantity(85.1, 'mV')
        thisSegmentData['conductance_IV'] = thisSegmentData['peak_current_density_IV'] \
                                          / thisSegmentData['driving_force']

        thisSegmentData['peak_or_mean_IV'] = 'mean' if thisSegmentData['voltage'] <= cellDetails['activation_voltage'] else 'peak'

        thisSegmentData['selected_current_IV']         = thisSegmentData[thisSegmentData['peak_or_mean_IV'] + '_current_IV']
        thisSegmentData['selected_current_density_IV'] = thisSegmentData[thisSegmentData['peak_or_mean_IV'] + '_current_density_IV']

        # Choose whether to use peak or mean for SSI curve, depending on the inactivation voltage
        thisSegmentData['peak_or_mean_SSI'] = 'peak' if thisSegmentData['voltage'] <= cellDetails['inactivation_voltage'] else 'mean'
        thisSegmentData['selected_current_SSI']         = thisSegmentData[thisSegmentData['peak_or_mean_SSI'] + '_current_SSI']

        # Define first_segment_peak_current_SSI
        if first_segment_peak_current_SSI is None:
          first_segment_peak_current_SSI = thisSegmentData['peak_current_SSI']

        # Calculate normalised SSI current
        thisSegmentData['normalised_current_SSI']       = thisSegmentData['selected_current_SSI']/first_segment_peak_current_SSI

        if (thisSegmentData['peak_current_IV'] < perCellMinPeakSoFar):
          perCellMinPeakSoFar = thisSegmentData['peak_current_IV']

        if (perCellMaxConductanceSoFar < thisSegmentData['conductance_IV']):
          perCellMaxConductanceSoFar = thisSegmentData['conductance_IV']

        perCellRunningTotalP2PNoise += thisSegmentData['peakNoisePos'].item() - thisSegmentData['peakNoiseNeg'].item()
        perCellRunningTotalRMSNoise += thisSegmentData['rmsNoise']

      cellDetails['min_peak_current_IV'] = perCellMinPeakSoFar
      cellDetails['max_conductance']  = perCellMaxConductanceSoFar
      cellDetails['mean_p2p_noise']   = perCellRunningTotalP2PNoise / segment_count
      cellDetails['mean_rms_noise']   = perCellRunningTotalRMSNoise / segment_count

      for thisSegmentData in cellDetails['segments']:
        thisSegmentData['normalised_conductance'] = thisSegmentData['conductance_IV'] / cellDetails['max_conductance']

      # Write the per-cell results
      percellfile.write('\t'.join([cellDetails['path']
                                  ,cellDetails['filename']
                                  ,cellDetails['experiment']
                                  ,cellDetails['cell_line']
                                  ,cellDetails['cell_source']
                                  ,cellDetails['freshness']
                                  ,cellDetails['cell_identity']
                                  ,cellDetails['when']
                                  ,str(cellDetails['whole_cell_capacitance'].item())
                                  ,str(cellDetails['min_peak_current_IV'].item())
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
                                     ,cellDetails['when']
                                     ,str(thisSegmentData['segmentIndex'])
                                     ,str(thisSegmentData['voltage'].item())
                                     ,thisSegmentData['peak_or_mean_IV']
                                     ,str(thisSegmentData['driving_force'].item())
                                     ,str(thisSegmentData['time_to_peak_IV'].item())
                                     ,str(thisSegmentData['peak_current_IV'].item())
                                     ,str(thisSegmentData['mean_current_IV'].item())
                                     ,str(thisSegmentData['selected_current_IV'].item())
                                     ,str(cellDetails['whole_cell_capacitance'].item())
                                     ,str(thisSegmentData['peak_current_density_IV'].item())
                                     ,str(thisSegmentData['mean_current_density_IV'].item())
                                     ,str(thisSegmentData['selected_current_density_IV'].item())
                                     ,str(thisSegmentData['conductance_IV'].item())
                                     ,str(cellDetails['max_conductance'].item())
                                     ,str(thisSegmentData['normalised_conductance'].item())
                                     ,cellDetails['classification']
                                     ,str(thisSegmentData['peakNoiseNeg'].item())
                                     ,str(thisSegmentData['peakNoisePos'].item())
                                     ,str(thisSegmentData['peak_current_SSI'].item())
                                     ,str(thisSegmentData['mean_current_SSI'].item())
                                     ,str(thisSegmentData['selected_current_SSI'].item())
                                     ,str(thisSegmentData['normalised_current_SSI'].item())
                                     ]) + '\n')

      # Draw the IV traces for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Time (s)')
      plt.ylabel('I (pA)')
      axes = plt.gca()
      axes.set_ylim([-670,100])

      for thisSegmentData in cellDetails['segments']:
        thisSegmentColor = colormap.to_rgba(segment_count - thisSegmentData['segmentIndex'])

        line = plt.plot([tPlotIVFrom + sample_index * sample_time_sec
                            for sample_index in range(len(thisSegmentData['traceToDraw_IV']))],
                        thisSegmentData['traceToDraw_IV'], linewidth=0.5, alpha=0.5)
        plt.setp(line, color=thisSegmentColor)

        if thisSegmentData['peak_or_mean_IV'] == 'peak':
          mark = plt.plot(thisSegmentData['time_to_peak_IV'], thisSegmentData['peak_current_IV'], '+')
          plt.setp(mark, color=thisSegmentColor)

      plt.axvspan(tAnalyseIVFrom, tAnalyseIVTo, facecolor='#c0c0c0', alpha=0.5)
      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'iv-traces-' + cellDetails['filename'] + ".png"))
      plt.close()
      
      # Draw the SSI traces for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Time (s)')
      plt.ylabel('I (pA)')
      axes = plt.gca()
      axes.set_ylim([-1000,100])

      for thisSegmentData in cellDetails['segments']:
        thisSegmentColor = colormap.to_rgba(segment_count - thisSegmentData['segmentIndex'])

        line = plt.plot([tPlotSSIFrom + sample_index * sample_time_sec
                            for sample_index in range(len(thisSegmentData['traceToDraw_SSI']))],
                        thisSegmentData['traceToDraw_SSI'], linewidth=0.5, alpha=0.5)
        plt.setp(line, color=thisSegmentColor)

        if thisSegmentData['peak_or_mean_SSI'] == 'peak':
          mark = plt.plot(thisSegmentData['time_to_peak_SSI'], thisSegmentData['peak_current_SSI'], '+')
          plt.setp(mark, color=thisSegmentColor)

      plt.axvspan(tAnalyseSSIFrom, tAnalyseSSITo, facecolor='#c0c0c0', alpha=0.5)
      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'ssi-traces-' + cellDetails['filename'] + ".png"))
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
        yData.append(thisSegmentData['peak_current_density_IV'])

      line = plt.plot(xData, yData, linewidth=0.5, zorder=0)
      plt.setp(line, color='#ff0000')

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['mean_current_density_IV'])

      line = plt.plot(xData, yData, linewidth=0.5, zorder=0)
      plt.setp(line, color='#00ff00')


      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['selected_current_density_IV'])

      line = plt.plot(xData, yData, linewidth=1, zorder=1)
      plt.setp(line, color='#000000')

      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'current-density-' + cellDetails['filename'] + '.png'))
      plt.close()

      # Draw the SSI curve for this cell
      figure = plt.figure(figsize=(20,10), dpi=80)
      plt.title(sampleName)
      plt.xlabel('Voltage (mV)')
      plt.ylabel('Normalised Current')

      xData = []
      yData = []
      for thisSegmentData in cellDetails['segments']:
        xData.append(thisSegmentData['voltage'])
        yData.append(thisSegmentData['normalised_current_SSI'])

      line = plt.plot(xData, yData, linewidth=0.5, zorder=0)
      plt.setp(line, color='#ff0000')
      plt.axvspan(tAnalyseSSIFrom, tAnalyseSSITo, facecolor='#c0c0c0', alpha=0.5)
      
      plt.grid()
      plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'inactivation-' + cellDetails['filename'] + '.png'))
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
        yData.append(thisSegmentData['peak_current_density_IV'])

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
        yData.append(thisSegmentData['selected_current_density_IV'])

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

    plt.grid()
    plt.savefig(os.path.join(resultsDirectory, experiment, condition, 'normalised-conductance-all.png'))
    plt.close()

pertracefile.close()
percellfile.close()

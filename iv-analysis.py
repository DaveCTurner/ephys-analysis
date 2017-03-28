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

      # Convert WCC to a number, but only if it's not blank
      wccString = cellDetailsRow['whole_cell_capacitance']
      wccVal    = float(wccString) if wccString else None

      cellDetailsByCell[cellDetailsRow['filename']] = \
        { 'filename':               cellDetailsRow['filename']       \
        , 'path':                   cellDetailsRow['path']           \
        , 'whole_cell_capacitance': wccVal                           \
        , 'cell_line':              cellDetailsRow['cell_line']      \
        , 'cell_source':            cellDetailsRow['cell_source']    \
        , 'protocol':               cellDetailsRow['protocol']       \
        , 'freshness':              cellDetailsRow['freshness']      \
        , 'classification':         cellDetailsRow['classification'] \
        , 'date':                   cellDetailsRow['date']           \
        , 'notes':                  cellDetailsRow['notes']          \
        }

# Define colour map: 'winter' is kinda green to kinda blue.
cmap = cm.get_cmap('winter')

# Define time points for the analysis
tStart          = 0.245  # Start the graph and start estimating the baseline
tBaselineLength = 0.01   # Estimate the baseline for this long
tEnd            = 0.268  # End the graph
tAnalyseFrom    = 0.2557 # Look for peaks after this time
tAnalyseTo      = 0.263  # Look for peaks before this time

filenames = glob(args.path + '\\**\\*.abf', recursive=True)

# Open a results file with the date in the filename
rundate = datetime.datetime.utcnow().replace(microsecond=0) \
                          .isoformat('-').replace(':','-')

pertracefilename = rundate + '-results-per-trace.txt'
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

percellfilename = rundate + '-results-per-cell.txt'
print ("Writing per-cell results to", percellfilename)
percellfile = open(percellfilename, 'w')
percellfile.write('Filename\tBest peak (pA)\tMean RMS noise (pA)\tMean P2P noise\n')

def voltageFromSegmentIndex(segmentIndex):
  return (5 * segmentIndex - 85)

for filename in filenames:
    cellDetails = cellDetailsByCell.get(basename(filename), None)
    if (cellDetails == None):
      print("No cell details found for", filename)
      continue

    if (cellDetails['protocol'] != 'IV'):
      continue

    sampleName = filename[len(args.path):]

    print ("Processing", sampleName)

    # Read the file into 'blocks'
    reader = AxonIO(filename=filename)
    blocks = reader.read()

    # Create a new graph
    fig = plt.figure(figsize=(20,10),dpi=80)
    plt.title(sampleName)
    plt.xlabel('Time (s)')
    plt.ylabel('I (pA)')
    axes = plt.gca()
    axes.set_ylim([-670,100])

    # Count how many segments there are and set up the colormap accordingly
    segmentIndex = 0
    segmentCount = len(blocks[0].segments)
    norm = mpl.colors.Normalize(vmin=1, vmax=segmentCount)
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Per-cell statistics
    perCellMinPeak       = 0
    perCellTotalP2PNoise = 0
    perCellTotalRMSNoise = 0

    for seg in blocks[0].segments:
        segmentIndex = segmentIndex + 1

        # Use the corrected signal [1] as opposed to the uncorrected [0]
        signal = seg.analogsignals[1]

        # Calculate the length of each sample in seconds
        sample_time_sec = (pq.Quantity(1, 'Hz') / signal.sampling_rate).item()

        # Estimate the baseline
        baseline = mean(signal[tStart / sample_time_sec : (tStart + tBaselineLength) / sample_time_sec])

        # Only take the signal from tStart to tEnd and take out the estimated baseline
        offsetted_I = signal[tStart / sample_time_sec : tEnd / sample_time_sec] - baseline

        # Find the +ve and -ve peak noise values
        quiescentSignal = offsetted_I[:tBaselineLength / sample_time_sec]
        peakNoiseNeg = min(quiescentSignal)
        peakNoisePos = max(quiescentSignal)
        meanSquareNoise = mean(quiescentSignal**2)
        meanSquareNoise.units = 'pA**2'
        rmsNoise = sqrt(meanSquareNoise)

        # Draw the exact signal
        line = plt.plot(arange(tStart + tBaselineLength, tAnalyseTo, sample_time_sec),
                 offsetted_I[tBaselineLength / sample_time_sec : (tAnalyseTo - tStart) / sample_time_sec],
				 linewidth=0.5, alpha=0.48)
        color = colormap.to_rgba(segmentCount - segmentIndex)
        plt.setp(line, color=color)

        # Just take the bit of the signal that needs searching for the peak
        toAnalyse = offsetted_I [ (tAnalyseFrom - tStart) / sample_time_sec
                              : (tAnalyseTo   - tStart) / sample_time_sec
                              ]

        # Find the peak index (number of samples), current and time
        minIndex = argmin(toAnalyse)
        minCurrent = toAnalyse[minIndex]
        minTime = minIndex * sample_time_sec + tAnalyseFrom

        # Draw a mark at the peak
        mark = plt.plot(minTime, minCurrent, '+')
        plt.setp(mark, color=color)

        minCurrent.units = 'pA'
        peakNoiseNeg.units = 'pA'
        peakNoisePos.units = 'pA'

        # Write the position of the peak to the results file
        pertracefile.write('\t'.join(
          [ sampleName
          , cellDetails['filename']
          , cellDetails['cell_line']
          , cellDetails['cell_source']
          , cellDetails['freshness']
          , str(segmentIndex)
          , str(voltageFromSegmentIndex(segmentIndex))
          , str(voltageFromSegmentIndex(segmentIndex) - 85.1)
          , str(minTime)
          , str(minCurrent.item())
          , str(cellDetails['whole_cell_capacitance'])
          , str(minCurrent.item() / cellDetails['whole_cell_capacitance'])
          , cellDetails['classification']
          , str(peakNoiseNeg.item())
          , str(peakNoisePos.item())
          ]) + '\n')

        if (minCurrent.item() < perCellMinPeak):
          perCellMinPeak = minCurrent.item()

        perCellTotalP2PNoise += peakNoisePos.item() - peakNoiseNeg.item()
        perCellTotalRMSNoise += rmsNoise

    percellfile.write(sampleName                     + '\t'
      + str(perCellMinPeak)                          + '\t'
      + str(perCellTotalRMSNoise / segmentCount)     + '\t'
      + str(perCellTotalP2PNoise / segmentCount)     + '\n')

    # Shade the part of the graph where the peak was sought
    plt.axvspan(tAnalyseFrom, tAnalyseTo, facecolor='#c0c0c0', alpha=0.5)

    # Save the graph next to the data file
    plt.grid()
    plt.savefig(filename + '-iv-graph.png')
    plt.close()

pertracefile.close()
percellfile.close()

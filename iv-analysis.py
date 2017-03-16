#!/usr/bin/python
# coding: utf-8

from neo.io import AxonIO
from numpy import mean, std, arange, convolve, ones, amin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from math import floor
import quantities as pq
from glob import glob
import datetime

# Define the moving average window: '0.02' repeated 50 times.
window = ones(50)/50.0

# Define colour map: 'winter' is kinda green to kinda blue.
cmap = cm.get_cmap('winter')

# Define time points for the analysis
tStart          = 0.245  # Start the graph and start estimating the baseline
tBaselineLength = 0.01   # Estimate the baseline for this long
tEnd            = 0.268  # End the graph
tAnalyseFrom    = 0.2562 # Look for peaks after this time
tAnalyseTo      = 0.263  # Look for peaks before this time

filenames = glob('data/iv/*.abf')

# Open a results file with the date in the filename
resultsfilename = 'results-' + \
                  datetime.datetime.utcnow().replace(microsecond=0) \
                          .isoformat('-').replace(':','-') + '.txt'
print ("Writing results to", resultsfilename)
resultsfile = open(resultsfilename, 'w')
resultsfile.write('Filename\tsegment\tt_min(s)\tI_min(pA)\n')

for filename in filenames:
    print ("Processing", filename)
    
    # Read the file into 'blocks'
    reader = AxonIO(filename=filename)
    blocks = reader.read()

    # Create a new graph
    fig = plt.figure(figsize=(10,5),dpi=80)
    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel('I (pA)')

    # Count how many segments there are and set up the colormap accordingly
    segmentIndex = 0
    segmentCount = len(blocks[0].segments)
    norm = mpl.colors.Normalize(vmin=1, vmax=segmentCount)
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)

    for seg in blocks[0].segments:
        segmentIndex = segmentIndex + 1

        # Use the corrected signal [1] as opposed to the uncorrected [0]
        signal = seg.analogsignals[1]

        # Calculate the length of each sample in seconds
        sample_time_sec = (pq.Quantity(1, 'Hz') / signal.sampling_rate).item()

        # Estimate the baseline
        baseline = mean(signal[tStart / sample_time_sec : (tStart + tBaselineLength) / sample_time_sec])

        # Only take the signal from tStart to tEnd and take out the estimated baseline
        offsetted = signal[tStart / sample_time_sec : tEnd / sample_time_sec] - baseline

        # Apply the moving average
        smoothed = convolve(offsetted, window, 'same')

        # Draw the exact signal faintly (alpha = 0.08)
        line = plt.plot(arange(tStart, tEnd, sample_time_sec),
                 offsetted, linewidth=0.5, alpha=0.08)
        color = colormap.to_rgba(segmentCount - segmentIndex)
        plt.setp(line, color=color)

        # Draw the smoothed signal
        line = plt.plot(arange(tStart, tEnd, sample_time_sec),
                 smoothed, linewidth=0.5, alpha=0.48)
        color = colormap.to_rgba(segmentCount - segmentIndex)
        plt.setp(line, color=color)

        # Just take the bit of the smoothed signal that needs searching for the peak
        toAnalyse = smoothed [ (tAnalyseFrom - tStart) / sample_time_sec
                             : (tAnalyseTo   - tStart) / sample_time_sec
                             ]

        # Find the peak index (number of samples), current and time
        minIndex = toAnalyse.argmin()
        minCurrent = toAnalyse[minIndex]
        minTime = minIndex * sample_time_sec + tAnalyseFrom

        # Draw a mark at the peak
        mark = plt.plot(minTime, minCurrent, '+')
        plt.setp(mark, color=color)

        # Write the position of the peak to the results file
        resultsfile.write(filename          + '\t'
                        + str(segmentIndex) + '\t'
                        + str(minTime)      + '\t'
                        + str(minCurrent)   + '\n')

    # Shade the part of the graph where the baseline was calculated
    plt.axvspan(tStart, tStart + tBaselineLength, facecolor='#808080', alpha=0.5)

    # Shade the part of the graph where the peak was sought
    plt.axvspan(tAnalyseFrom, tAnalyseTo, facecolor='#c0c0c0', alpha=0.5)

    # Save the graph next to the data file
    plt.grid()
    plt.savefig(filename + '.png')

resultsfile.close()

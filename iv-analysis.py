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

window = ones(50)/50.0
cmap = cm.get_cmap('winter')

tStart = 0.245
tBaselineLength = 0.01
tEnd   = 0.268
tAnalyseFrom = 0.2562
tAnalyseTo   = 0.263

filenames = glob('data/iv/*.abf')
resultsfile = open('results.txt', 'w')
resultsfile.write('Filename\tsegment\tt_min(s)\tI_min(pA)\n')

for filename in filenames:
    reader = AxonIO(filename=filename)
    blocks = reader.read()

    fig = plt.figure(figsize=(10,3),dpi=80)
    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel('I (pA)')

    segmentIndex = 0
    segmentCount = len(blocks[0].segments)
    norm = mpl.colors.Normalize(vmin=1, vmax=segmentCount)
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)

    for seg in blocks[0].segments:
        segmentIndex = segmentIndex + 1

        signal = seg.analogsignals[1]
        sample_time_sec = (pq.Quantity(1, 'Hz') / signal.sampling_rate).item()

        baseline = mean(signal[tStart / sample_time_sec : (tStart + tBaselineLength) / sample_time_sec])

        offsetted = signal[tStart / sample_time_sec : tEnd / sample_time_sec] - baseline

        smoothed = convolve(offsetted, window, 'same')

        line = plt.plot(arange(tStart, tEnd, sample_time_sec),
                 offsetted, linewidth=0.5, alpha=0.08)
        color = colormap.to_rgba(segmentCount - segmentIndex)
        plt.setp(line, color=color)

        line = plt.plot(arange(tStart, tEnd, sample_time_sec),
                 smoothed, linewidth=0.5, alpha=0.48)
        color = colormap.to_rgba(segmentCount - segmentIndex)
        plt.setp(line, color=color)

        toAnalyse = smoothed [ (tAnalyseFrom - tStart) / sample_time_sec
                             : (tAnalyseTo   - tStart) / sample_time_sec
                             ]

        minIndex = toAnalyse.argmin()
        minCurrent = toAnalyse[minIndex]
        minTime = minIndex * sample_time_sec + tAnalyseFrom
        mark = plt.plot(minTime, minCurrent, '+')
        plt.setp(mark, color=color)
        resultsfile.write(filename          + '\t'
                        + str(segmentIndex) + '\t'
                        + str(minTime)      + '\t'
                        + str(minCurrent)   + '\n')

    plt.axvspan(tStart, tStart + tBaselineLength, facecolor='#808080', alpha=0.5)
    plt.axvspan(tAnalyseFrom, tAnalyseTo, facecolor='#c0c0c0', alpha=0.5)
    plt.grid()
    plt.savefig(filename + '.png')

resultsfile.close()

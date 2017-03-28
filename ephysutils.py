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

def loadCellDetails(cellDetailsFilename):
  # Load cell-details.txt
  cellDetailsByCell = {}
  with open(cellDetailsFilename) as cellDetailsFile:
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
  return cellDetailsByCell

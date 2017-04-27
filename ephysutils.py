import csv
import datetime
import os
import glob
import quantities as pq

def loadCellDetails(cellDetailsFilename):
  # Load cell-details.txt
  cellDetailsByCell = {}
  with open(cellDetailsFilename) as cellDetailsFile:
    cellDetailsReader = csv.DictReader(cellDetailsFile, delimiter='\t')
    for cellDetailsRow in cellDetailsReader:
      if (cellDetailsRow['path'] != ''):

        # Convert WCC to a number, but only if it's not blank
        wccString = cellDetailsRow['whole_cell_capacitance']
        wccVal    = pq.Quantity(float(wccString), 'pF') if wccString else None

        acVString = cellDetailsRow['activation_voltage']
        acVVal    = pq.Quantity(float(acVString), 'mV') if acVString else None

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
          , 'experiment':             cellDetailsRow['experiment']     \
          , 'activation_voltage':     acVVal                           \
          , 'cell_identity':          cellDetailsRow['cell_identity']  \
          }

  return cellDetailsByCell

def makeResultsDirectory():
  rundate = datetime.datetime.utcnow().replace(microsecond=0) \
                          .isoformat('-').replace(':','-')
  dirname = os.path.join('results', rundate)
  os.makedirs(dirname)
  return dirname

# Find trace files and organise them into an experiment/protocol/condition tree;
# e.g. traceFiles['coverslip']['IV']['Primary 231-GFP fresh']
#       = [ {'filename': <FULL FILE NAME>
#           ,'details':  <CELL DETAILS>
#           } ]
def findTraceFiles(searchRoot, cellDetailsByCell):
  filenames = glob.glob(os.path.join(searchRoot, '**', '*.abf'), recursive=True)

  traceFiles = {}

  for filename in filenames:
    cellDetails = cellDetailsByCell.get(os.path.basename(filename), None)
    if (cellDetails == None):
      print("No cell details for", filename)
      continue

    thisExperiment = traceFiles.get(cellDetails['experiment'], None)
    if thisExperiment == None:
      thisExperiment = {}
      traceFiles[cellDetails['experiment']] = thisExperiment

    thisProtocol = thisExperiment.get(cellDetails['protocol'], None)
    if thisProtocol == None:
      thisProtocol = {}
      thisExperiment[cellDetails['protocol']] = thisProtocol

    condition = " ".join([cellDetails['cell_line'], cellDetails['cell_source'], cellDetails['freshness']]).strip()

    thisCondition = thisProtocol.get(condition, None)
    if (thisCondition == None):
      thisCondition = {'files':[]}
      thisProtocol[condition] = thisCondition

    thisCondition['files'].append({'filename':filename, 'details':cellDetails})

  return traceFiles
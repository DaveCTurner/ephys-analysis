import csv

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

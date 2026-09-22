"""Loading of example data: one float array per sample set, the time column at index 0."""

import importlib.resources  # Get data file pathnames from EDM package

from pandas import read_csv, to_numeric

dataFileNames = [ ("TentMap.csv",             "TentMap"),
                  ("TentMapNoise.csv",        "TentMapNoise"),
                  ("circle.csv",              "circle"),
                  ("circle_noise.csv",        "circleNoise"),
                  ("circle_noTime.csv",       "circle_noTime"),
                  ("columnNameSpace.csv",     "columnNameSpace"),
                  ("block_3sp.csv",           "block_3sp"),
                  ("sardine_anchovy_sst.csv", "sardine_anchovy_sst"),
                  ("LorenzData1000.csv",      "Lorenz5D"),
                  ("S12CD-S333-SumFlow_1980-2005.csv", "SumFlow_1980-2005") ]

# name -> [nRows, nColumns] float array; column 0 is the time column, which the functions never take
sampleData = {}

for fileName, dataName in dataFileNames:

    filePath = "data/" + fileName

    ref = importlib.resources.files('torchEDM') / filePath

    with importlib.resources.as_file( ref ) as filePath_ :
        # a date-valued time column becomes NaN; the functions never take column 0
        sampleData[ dataName ] = read_csv( filePath_ ).apply( to_numeric, errors = 'coerce' ).values.astype( float )

if not len( sampleData ) :
    raise Warning( "torchEDM: Failed to find sample data in torchEDM package." )

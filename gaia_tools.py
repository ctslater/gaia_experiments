
import os
import healpy as hp
import numpy as np
import pandas as pd
import pyspark.sql.functions as sparkfunc

def spark_start(project_path, metastore=None, processes=8):
    from pyspark.sql import SparkSession

    warehouse_location = os.path.join(project_path, 'spark-warehouse')

    local_dir = os.path.join(project_path, 'spark-tmp')

    spark = ( 
            SparkSession.builder
            .appName("LSD2")
            .config("spark.sql.warehouse.dir", warehouse_location)
            .config('spark.master', "local[%d]" % processes)
            .config('spark.driver.memory', '16G') # 128
            .config('spark.local.dir', local_dir)
            .config('spark.memory.offHeap.enabled', 'true')
            .config('spark.memory.offHeap.size', '16G') # 256
            .config("spark.sql.execution.arrow.enabled", "true")
            .config("spark.driver.maxResultSize", "8G")
            .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={metastore}")
            .enableHiveSupport()
            .getOrCreate()
                    )   

    return spark

def healpix_hist(input_df, NSIDE=64, groupby=[],
                 agg={"*": "count"}, returnDf=False):
    from pyspark.sql.functions import floor as FLOOR, col as COL, lit, shiftRight

    order0 = 12
    order  = hp.nside2order(NSIDE)
    shr    = 2*(order0 - order)

    # construct query
    df = input_df.withColumn('hpix__', shiftRight('hpix12', shr))
    gbcols = ('hpix__', )
    for axspec in groupby:
        if not isinstance(axspec, str):
            (col, c0, c1, dc) = axspec
            df = ( df
                .where((lit(c0) < COL(col)) & (COL(col) < lit(c1)))
                .withColumn(col + '_bin__', FLOOR((COL(col) - lit(c0)) / lit(dc)) * lit(dc) + lit(c0) )
                 )
            gbcols += ( col + '_bin__', )
        else:
            gbcols += ( axspec, )
    df = df.groupBy(*gbcols)

    # execute aggregation
    df = df.agg(agg)

    # fetch result
    df = df.toPandas()
    if returnDf:
        return df

    # repack the result into maps
    # This results line is slightly dangerous, because some aggregate functions are purely aliases.
    # E.g., mean(x) gets returned as a column avg(x).
    results = [ f"{v}({k})" if k != "*" else f"{v}(1)" for k, v in agg.items() ]    # Result columns
    def _create_map(df):
        maps = dict()
        for val in results:
            map_ = np.zeros(hp.nside2npix(NSIDE))
            # I think this line throws an error if there are no rows in the result
            map_[df.hpix__.values] = df[val].values 
            maps[val] = [ map_ ]
        return pd.DataFrame(data=maps)

    idxcols = list(gbcols[1:])
    if len(idxcols) == 0:
        ret = _create_map(df)
        assert(len(ret) == 1)
        if not returnDf:
            # convert to tuple, or scalar
            ret = tuple(ret[name].values[0] for name in results)
            if len(ret) == 1:
                ret = ret[0]
    else:
        ret = df.groupby(idxcols).apply(_create_map)
        ret.index = ret.index.droplevel(-1)
        ret.index.rename([ name.split("_bin__")[0] for name in ret.index.names ], inplace=True)
        if "count(1)" in ret:
                    ret = ret.rename(columns={'count(1)': 'count'})
        if not returnDf:
            if len(ret.columns) == 1:
                ret = ret.iloc[:, 0]
    return ret



def bin_column(start, stop, bins, data):
    bin_size = (stop - start)/bins
    return sparkfunc.floor((data - start)/bin_size)
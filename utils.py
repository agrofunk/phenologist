from glob import glob
import xarray as xr
import rasterio as rio
import warnings
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import boto3


def open_ncs(folder, pattern):
    ff = sorted(glob(f'{folder}/{pattern}*.nc'))

    df = xr.open_dataset(ff[0], chunks=dict(time=-1))
    print(df.dims)
    df = df.rio.write_crs('epsg:4326', inplace=True)
    df = df.rename({"longitude": "x", "latitude": "y"})
    for f in ff[1:]:
        with xr.open_dataset(f) as df1:
            print(df1.dims)
            df1 = df1.rio.write_crs('epsg:4326', inplace=True)
            df1 = df1.rio.reproject_match(df)
        df = xr.concat([df, df1], dim='time', coords='minimal', compat='override', data_vars='all')
        del df1
    df = df.rename({"x": "longitude", "y": "latitude"})
    return df


def clidev(ds,
           month=True,
           plot=True,
           pcs=(1, 99)  # just for plotting
           ):
    ''' 
        return zscore, anomalies, mean, stdev
    '''

    me = ds.groupby(ds.time.dt.month).mean()
    sd = ds.groupby(ds.time.dt.month).std().clip(min=0.001)

    concat_z = []
    concat_a = []
    for m in range(1, 13):
        # anomalies
        globals()[f'a{str(m).zfill(2)}'] = (ds.sel(time=ds.time.dt.month.isin([m])) - me.sel(month=m))
        # zscores
        globals()[f'z{str(m).zfill(2)}'] = globals()[f'a{str(m).zfill(2)}'] / sd.sel(month=m)
        concat_z.append(globals()[f'z{str(m).zfill(2)}'])
        concat_a.append(globals()[f'a{str(m).zfill(2)}'])
    df_z = xr.concat(concat_z, dim='time')
    df_z = df_z.sortby('time')
    df_a = xr.concat(concat_a, dim='time')
    df_a = df_a.sortby('time')

    if month:
        df_z = df_z.resample(time='ME').mean(skipna=True)
        df_a = df_a.resample(time='ME').mean(skipna=True)
    if plot:
        plt.hist(np.ravel(df_z), bins=200)
        plt.title(
            f'extreme values for zscores --> {np.nanpercentile(df_z, pcs[0]):.3f} {np.nanpercentile(df_z, pcs[1]):.3f}');
        plt.grid();
        plt.show();
        plt.close()
        plt.hist(np.ravel(df_a), bins=200)
        plt.title(
            f'extreme values for anomalies --> {np.nanpercentile(df_a, pcs[0]):.3f} {np.nanpercentile(df_a, pcs[1]):.3f}');
        plt.grid();
        plt.show();
        plt.close()

    return df_z, df_a, me, sd


def calculate_indices(ds,
                      index=None,
                      collection=None,
                      custom_varname=None,
                      normalise=True,
                      drop=False,
                      inplace=False):
    """
    Takes an xarray dataset containing spectral bands, calculates one of
    a set of remote sensing indices, and adds the resulting array as a 
    new variable in the original dataset.  
    
    Note: by default, this function will create a new copy of the data
    in memory. This can be a memory-expensive operation, so to avoid
    this, set `inplace=True`.

    Last modified: June 2023
    
    Parameters
    ----------
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array with containing the
        spectral bands required to calculate the index. These bands are
        used as inputs to calculate the selected water index.
    index : str or list of strs
        A string giving the name of the index to calculate or a list of
        strings giving the names of the indices to calculate:
        
        * ``'AWEI_ns'`` (Automated Water Extraction Index,
                  no shadows, Feyisa 2014)
        * ``'AWEI_sh'`` (Automated Water Extraction Index,
                   shadows, Feyisa 2014)
        * ``'BAEI'`` (Built-Up Area Extraction Index, Bouzekri et al. 2015)
        * ``'BAI'`` (Burn Area Index, Martin 1998)
        * ``'BSI'`` (Bare Soil Index, Rikimaru et al. 2002)
        * ``'BUI'`` (Built-Up Index, He et al. 2010)
        * ``'CMR'`` (Clay Minerals Ratio, Drury 1987)
        * ``'EVI'`` (Enhanced Vegetation Index, Huete 2002)
        * ``'FMR'`` (Ferrous Minerals Ratio, Segal 1982)
        * ``'IOR'`` (Iron Oxide Ratio, Segal 1982)
        * ``'LAI'`` (Leaf Area Index, Boegh 2002)
        * ``'MNDWI'`` (Modified Normalised Difference Water Index, Xu 1996)
        * ``'MSAVI'`` (Modified Soil Adjusted Vegetation Index,
                 Qi et al. 1994)              
        * ``'NBI'`` (New Built-Up Index, Jieli et al. 2010)
        * ``'NBR'`` (Normalised Burn Ratio, Lopez Garcia 1991)
        * ``'NDBI'`` (Normalised Difference Built-Up Index, Zha 2003)
        * ``'NDCI'`` (Normalised Difference Chlorophyll Index, 
                Mishra & Mishra, 2012)
        * ``'NDMI'`` (Normalised Difference Moisture Index, Gao 1996)        
        * ``'NDSI'`` (Normalised Difference Snow Index, Hall 1995)
        * ``'NDTI'`` (Normalise Difference Tillage Index,
                Van Deventeret et al. 1997)
        * ``'NDTI2'`` (Normalised Difference Turbidity Index, Lacaux et al., 2007)
        * ``'NDVI'`` (Normalised Difference Vegetation Index, Rouse 1973)
        * ``'NDWI'`` (Normalised Difference Water Index, McFeeters 1996)
        * ``'SAVI'`` (Soil Adjusted Vegetation Index, Huete 1988)
        * ``'TCB'`` (Tasseled Cap Brightness, Crist 1985)
        * ``'TCG'`` (Tasseled Cap Greeness, Crist 1985)
        * ``'TCW'`` (Tasseled Cap Wetness, Crist 1985)        
        * ``'TCB_GSO'`` (Tasseled Cap Brightness, Nedkov 2017)
        * ``'TCG_GSO'`` (Tasseled Cap Greeness, Nedkov 2017)
        * ``'TCW_GSO'`` (Tasseled Cap Wetness, Nedkov 2017)
        * ``'WI'`` (Water Index, Fisher 2016)
        * ``'kNDVI'`` (Non-linear Normalised Difference Vegation Index,
                 Camps-Valls et al. 2021)

    collection : str
        An string that tells the function what data collection is 
        being used to calculate the index. This is necessary because 
        different collections use different names for bands covering 
        a similar spectra. 
        
        Valid options are: 
        
        * ``'ga_ls_3'`` (for GA Landsat Collection 3) 
        * ``'ga_s2_3'`` (for GA Sentinel 2 Collection 3)
        * ``'ga_gm_3'`` (for GA Geomedian Collection 3)

    custom_varname : str, optional
        By default, the original dataset will be returned with 
        a new index variable named after `index` (e.g. 'NDVI'). To 
        specify a custom name instead, you can supply e.g. 
        `custom_varname='custom_name'`. Defaults to None, which uses
        `index` to name the variable. 
    normalise : bool, optional
        Some coefficient-based indices (e.g. ``'WI'``, ``'BAEI'``,
        ``'AWEI_ns'``, ``'AWEI_sh'``, ``'TCW'``, ``'TCG'``, ``'TCB'``, 
        ``'TCW_GSO'``, ``'TCG_GSO'``, ``'TCB_GSO'``, ``'EVI'``, 
        ``'LAI'``, ``'SAVI'``, ``'MSAVI'``) produce different results if 
        surface reflectance values are not scaled between 0.0 and 1.0 
        prior to calculating the index. Setting `normalise=True` first 
        scales values to a 0.0-1.0 range by dividing by 10000.0. 
        Defaults to True.  
    drop : bool, optional
        Provides the option to drop the original input data, thus saving 
        space. if drop = True, returns only the index and its values.
    inplace: bool, optional
        If `inplace=True`, calculate_indices will modify the original
        array in-place, adding bands to the input dataset. The default
        is `inplace=False`, which will instead make a new copy of the
        original data (and use twice the memory).
        
    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a 
        new varible containing the remote sensing index as a DataArray.
        If drop = True, the new variable/s as DataArrays in the 
        original Dataset. 
    """

    # Set ds equal to a copy of itself in order to prevent the function 
    # from editing the input dataset. This can prevent unexpected
    # behaviour though it uses twice as much memory.    
    if not inplace:
        ds = ds.copy(deep=True)

    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop = list(ds.data_vars)
        # print(f'Dropping bands {bands_to_drop}')

    # Dictionary containing remote sensing index band recipes
    index_dict = {
        # Normalised Difference Vegation Index, Rouse 1973
        'NDVI': lambda ds: (ds.nir - ds.red) /
                           (ds.nir + ds.red),

        # Non-linear Normalised Difference Vegation Index,
        # Camps-Valls et al. 2021
        'kNDVI': lambda ds: np.tanh(((ds.nir - ds.red) /
                                     (ds.nir + ds.red)) ** 2),

        # Enhanced Vegetation Index, Huete 2002
        'EVI': lambda ds: ((2.5 * (ds.nir - ds.red)) /
                           (ds.nir + 6 * ds.red -
                            7.5 * ds.blue + 1)),

        # Leaf Area Index, Boegh 2002
        'LAI': lambda ds: (3.618 * ((2.5 * (ds.nir - ds.red)) /
                                    (ds.nir + 6 * ds.red -
                                     7.5 * ds.blue + 1)) - 0.118),

        # Soil Adjusted Vegetation Index, Huete 1988
        'SAVI': lambda ds: ((1.5 * (ds.nir - ds.red)) /
                            (ds.nir + ds.red + 0.5)),

        # Mod. Soil Adjusted Vegetation Index, Qi et al. 1994
        'MSAVI': lambda ds: ((2 * ds.nir + 1 -
                              ((2 * ds.nir + 1) ** 2 -
                               8 * (ds.nir - ds.red)) ** 0.5) / 2),

        # Normalised Difference Moisture Index, Gao 1996
        'NDMI': lambda ds: (ds.nir - ds.swir1) /
                           (ds.nir + ds.swir1),

        # Normalised Burn Ratio, Lopez Garcia 1991
        'NBR': lambda ds: (ds.nir - ds.swir2) /
                          (ds.nir + ds.swir2),

        # Burn Area Index, Martin 1998
        'BAI': lambda ds: (1.0 / ((0.10 - ds.red) ** 2 +
                                  (0.06 - ds.nir) ** 2)),

        # Normalised Difference Chlorophyll Index,
        # (Mishra & Mishra, 2012)
        'NDCI': lambda ds: (ds.red_edge_1 - ds.red) /
                           (ds.red_edge_1 + ds.red),

        # Normalised Difference Snow Index, Hall 1995
        'NDSI': lambda ds: (ds.green - ds.swir1) /
                           (ds.green + ds.swir1),

        # Normalised Difference Tillage Index,
        # Van Deventer et al. 1997
        'NDTI': lambda ds: (ds.swir1 - ds.swir2) /
                           (ds.swir1 + ds.swir2),

        # Normalised Difference Turbidity Index,
        # Lacaux et al., 2007
        'NDTI2': lambda ds: (ds.red - ds.green) /
                            (ds.red + ds.green),

        # Normalised Difference Water Index, McFeeters 1996
        'NDWI': lambda ds: (ds.green - ds.nir) /
                           (ds.green + ds.nir),

        # Modified Normalised Difference Water Index, Xu 2006
        'MNDWI': lambda ds: (ds.green - ds.swir1) /
                            (ds.green + ds.swir1),

        # Normalised Difference Built-Up Index, Zha 2003
        'NDBI': lambda ds: (ds.swir1 - ds.nir) /
                           (ds.swir1 + ds.nir),

        # Built-Up Index, He et al. 2010
        'BUI': lambda ds: ((ds.swir1 - ds.nir) /
                           (ds.swir1 + ds.nir)) -
                          ((ds.nir - ds.red) /
                           (ds.nir + ds.red)),

        # Built-up Area Extraction Index, Bouzekri et al. 2015
        'BAEI': lambda ds: (ds.red + 0.3) /
                           (ds.green + ds.swir1),

        # New Built-up Index, Jieli et al. 2010
        'NBI': lambda ds: (ds.swir1 + ds.red) / ds.nir,

        # Bare Soil Index, Rikimaru et al. 2002
        'BSI': lambda ds: ((ds.swir1 + ds.red) -
                           (ds.nir + ds.blue)) /
                          ((ds.swir1 + ds.red) +
                           (ds.nir + ds.blue)),

        # Automated Water Extraction Index (no shadows), Feyisa 2014
        'AWEI_ns': lambda ds: (4 * (ds.green - ds.swir1) -
                               (0.25 * ds.nir * + 2.75 * ds.swir2)),

        # Automated Water Extraction Index (shadows), Feyisa 2014
        'AWEI_sh': lambda ds: (ds.blue + 2.5 * ds.green -
                               1.5 * (ds.nir + ds.swir1) -
                               0.25 * ds.swir2),

        # Water Index, Fisher 2016
        'WI': lambda ds: (1.7204 + 171 * ds.green + 3 * ds.red -
                          70 * ds.nir - 45 * ds.swir1 -
                          71 * ds.swir2),

        # Tasseled Cap Wetness, Crist 1985
        'TCW': lambda ds: (0.0315 * ds.blue + 0.2021 * ds.green +
                           0.3102 * ds.red + 0.1594 * ds.nir +
                           -0.6806 * ds.swir1 + -0.6109 * ds.swir2),

        # Tasseled Cap Greeness, Crist 1985
        'TCG': lambda ds: (-0.1603 * ds.blue + -0.2819 * ds.green +
                           -0.4934 * ds.red + 0.7940 * ds.nir +
                           -0.0002 * ds.swir1 + -0.1446 * ds.swir2),

        # Tasseled Cap Brightness, Crist 1985
        'TCB': lambda ds: (0.2043 * ds.blue + 0.4158 * ds.green +
                           0.5524 * ds.red + 0.5741 * ds.nir +
                           0.3124 * ds.swir1 + -0.2303 * ds.swir2),

        # Tasseled Cap Transformations with Sentinel-2 coefficients
        # after Nedkov 2017 using Gram-Schmidt orthogonalization (GSO)
        # Tasseled Cap Wetness, Nedkov 2017
        'TCW_GSO': lambda ds: (0.0649 * ds.blue + 0.2802 * ds.green +
                               0.3072 * ds.red + -0.0807 * ds.nir +
                               -0.4064 * ds.swir1 + -0.5602 * ds.swir2),

        # Tasseled Cap Greeness, Nedkov 2017
        'TCG_GSO': lambda ds: (-0.0635 * ds.blue + -0.168 * ds.green +
                               -0.348 * ds.red + 0.3895 * ds.nir +
                               -0.4587 * ds.swir1 + -0.4064 * ds.swir2),

        # Tasseled Cap Brightness, Nedkov 2017
        'TCB_GSO': lambda ds: (0.0822 * ds.blue + 0.136 * ds.green +
                               0.2611 * ds.red + 0.5741 * ds.nir +
                               0.3882 * ds.swir1 + 0.1366 * ds.swir2),

        # Clay Minerals Ratio, Drury 1987
        'CMR': lambda ds: (ds.swir1 / ds.swir2),

        # Ferrous Minerals Ratio, Segal 1982
        'FMR': lambda ds: (ds.swir / ds.nir),

        # Iron Oxide Ratio, Segal 1982
        'IOR': lambda ds: (ds.red / ds.blue),

    }

    # If index supplied is not a list, convert to list. This allows us to
    # iterate through either multiple or single indices in the loop below
    indices = index if isinstance(index, list) else [index]

    # Calculate for each index in the list of indices supplied (indexes)
    for index in indices:

        # Select an index function from the dictionary
        index_func = index_dict.get(str(index))

        # If no index is provided or if no function is returned due to an 
        # invalid option being provided, raise an exception informing user to 
        # choose from the list of valid options
        if index is None:

            raise ValueError(f"No remote sensing `index` was provided. Please "
                             "refer to the function \ndocumentation for a full "
                             "list of valid options for `index` (e.g. 'NDVI')")

        elif (index in ['WI', 'BAEI', 'AWEI_ns', 'AWEI_sh',
                        'EVI', 'LAI', 'SAVI', 'MSAVI']
              and not normalise):

            warnings.warn(f"\nA coefficient-based index ('{index}') normally "
                          "applied to surface reflectance values in the \n"
                          "0.0-1.0 range was applied to values in the 0-10000 "
                          "range. This can produce unexpected results; \nif "
                          "required, resolve this by setting `normalise=True`")

        elif index_func is None:

            raise ValueError(f"The selected index '{index}' is not one of the "
                             "valid remote sensing index options. \nPlease "
                             "refer to the function documentation for a full "
                             "list of valid options for `index`")

        # Rename bands to a consistent format if depending on what collection
        # is specified in `collection`. This allows the same index calculations
        # to be applied to all collections. If no collection was provided, 
        # raise an exception.
        if collection is None:

            raise ValueError("'No `collection` was provided. Please specify "
                             "either 'ga_ls_3', 'ga_s2_3' or 'ga_gm_3' "
                             "to ensure the function calculates indices "
                             "using the correct spectral bands")

        elif collection == 'Landsat':

            # Dictionary mapping full data names to simpler 'red' alias names
            bandnames_dict = {
                'nir08': 'nir',
                'red': 'red',
                'green': 'green',
                'blue': 'blue',
                'swir16': 'swir1',
                'swir22': 'swir2'
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        elif collection == 'ga_s2_3':

            # Dictionary mapping full data names to simpler 'red' alias names
            bandnames_dict = {
                'nbart_red': 'red',
                'nbart_green': 'green',
                'nbart_blue': 'blue',
                'nbart_nir_1': 'nir',
                'nbart_red_edge_1': 'red_edge_1',
                'nbart_red_edge_2': 'red_edge_2',
                'nbart_swir_2': 'swir1',
                'nbart_swir_3': 'swir2',
                'nbar_red': 'red',
                'nbar_green': 'green',
                'nbar_blue': 'blue',
                'nbar_nir_1': 'nir',
                'nbar_red_edge_1': 'red_edge_1',
                'nbar_red_edge_2': 'red_edge_2',
                'nbar_swir_2': 'swir1',
                'nbar_swir_3': 'swir2'
            }

            # Rename bands in dataset to use simple names (e.g. 'red')
            bands_to_rename = {
                a: b for a, b in bandnames_dict.items() if a in ds.variables
            }

        elif collection == 'ga_gm_3':

            # Pass an empty dict as no bands need renaming
            bands_to_rename = {}

        # Raise error if no valid collection name is provided:
        else:
            raise ValueError(f"'{collection}' is not a valid option for "
                             "`collection`. Please specify either \n"
                             "'Landsat', 'ga_s2_3' or 'ga_gm_3'")

        # Apply index function 
        try:
            # If normalised=True, divide data by 10,000 before applying func
            mult = 10000.0 if normalise else 1.0
            index_array = index_func(ds.rename(bands_to_rename) / mult)
        except AttributeError:
            raise ValueError(f'Please verify that all bands required to '
                             f'compute {index} are present in `ds`. \n'
                             f'These bands may vary depending on the `collection` '
                             f'(e.g. the Landsat `nbart_nir` band \n'
                             f'is equivelent to `nbart_nir_1` for Sentinel 2)')

        # Add as a new variable in dataset
        output_band_name = custom_varname if custom_varname else index
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if inplace=False
    if drop and not inplace:
        #ds = ds.drop(bands_to_drop)
        ds = ds.drop(set(bands_to_drop) - set(indices))
        print(f'Dropping bands {set(bands_to_drop) - set(indices)}')

    # If inplace == True, delete bands in-place instead of using drop
    if drop and inplace:
        for band_to_drop in bands_to_drop:
            del ds[band_to_drop]

    # Return input dataset with added water index variable
    return ds


def interp_z(z, days, zmin=None, zmax=None):
    if not zmin: zmin = -2.5
    if not zmax: zmax = 2.5

    zi = z.chunk(dict(time=-1))
    zi = xr.where((zi > zmax) | (zi < zmin), np.nan, zi)
    zi = zi.interpolate_na(dim='time',
                           method='pchip',  # pchip , spline
                           max_gap=np.timedelta64(days, 'D')
                           );
    return zi


def interp_anom(anom, days, amin=None, amax=None):
    if not amin: amin = np.nanpercentile(anom, 0.1)
    if not amax: amax = np.nanpercentile(anom, 99.9)
    anomi = anom.chunk(dict(time=-1))
    anomi = xr.where((anomi > amax) | (anomi < amin), np.nan, anomi)
    anomi = anomi.interpolate_na(dim='time',
                                 method='pchip',  # pchip , spline
                                 max_gap=np.timedelta64(days, 'D')
                                 );
    return anomi


def ds_cleanup(ds):
    '''
        drop all nan images AND
        drop images with near zero variance 
    '''
    import numpy

    # the simple
    ds = ds.dropna(dim='time', how='all')

    # to catch the mono valued arrays
    for t in ds.time.values:
        if numpy.isnan(numpy.nanvar(numpy.ravel(ds.sel(time=t)))) == True:
            ds = ds.drop([t], dim='time')
            print(f'{t} dropped due to all nan')

        # going further

        if numpy.nanvar(numpy.ravel(ds.sel(time=t))) < 0.0000001:
            ds = ds.drop([t], dim='time')
            print(f'{t} dropped due to very low variance')

    return ds


def png2color(path: str,
              cmap: str = 'RdYlBu',
              vmin: int | float = 1,  # 0
              vmax: int | float = 65535  # 65535
              ):
    '''
        inspired on
        https://stackoverflow.com/questions/8218608/savefig-without-frames-axes-only-content
    
    '''

    from imageio.v3 import imread
    plt.ioff()

    im = imread(path)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1], im.shape[0])  #0, 1
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im, cmap=cmap, aspect='auto',
              vmin=vmin, vmax=vmax);
    fig.savefig(path, dpi=1)
    plt.close(fig)
    del im, fig


def metadata_gdal_parser(metafile):
    import subprocess
    import re

    m_center_ = str(subprocess.check_output(['grep', "Center", metafile]))
    m_ur_ = str(subprocess.check_output(['grep', "Upper Right", metafile]))
    m_ll_ = str(subprocess.check_output(['grep', "Lower Left", metafile]))

    m_center = (re.split(r',|\s+|\)', m_center_)[2], re.split(r',|\s+|\)', m_center_)[4])
    m_ur = (re.split(r',|\s+|\)', m_ur_)[3], re.split(r',|\s+|\)', m_ur_)[5])
    m_ll = (re.split(r',|\s+|\)', m_ll_)[3], re.split(r',|\s+|\)', m_ll_)[5])

    print('returning 3 tuples: m_center, m_ur, m_ll')
    return m_center, m_ur, m_ll


def imageseries_monthly_png_s3(var,
                               farmgroupid: str,
                               bucketname: str,
                               vi: str,
                               satellite: str,
                               metric_type: str,
                               metric_type2: str,
                               bottom: int,
                               top: int,
                               datetime_range: str,
                               version: str,
                               desc: str,
                               run: str,
                               local_folder: str,
                               remove_local=True,
                               up: bool = True,
                               ):
    import os
    from pathlib import Path
    import numpy as np
    import boto3
    from botocore.exceptions import ClientError
    import logging
    from uuid import uuid4
    import rasterio as rio

    mapimages = {"L": []}
    today = str(np.datetime64('today'))
    folder = f'{local_folder}/{farmgroupid}/{today}/{metric_type}/{vi}/{metric_type2}/'
    s3folder = f'{farmgroupid}/{today}/{metric_type}/{vi}/{metric_type2}/'
    Path(folder).mkdir(parents=True, exist_ok=True)

    var = ds_cleanup(var)
    var = var.resample(time='ME').median(dim='time')
    dates = [str(x).split('T')[0] for x in var.time.values]

    # upload function
    def upload_file(filename, bucketname, objectname):
        s3_client = boto3.client('s3')

        try:
            response = s3_client.upload_file(filename,
                                             bucketname,
                                             objectname,
                                             ExtraArgs=themeta)

        except ClientError as e:
            logging.error(e)
            return False
        return True

    def add_s3path_to_mapimages(mapImages, path):

        mapImages['L'].append({"M": {'path': {'S': path}}})

    for i, date_ in enumerate(dates):
        uuid = str(uuid4())

        fn = f'{date_}_{uuid}'
        var.sel(time=date_, method='nearest').rio.to_raster(f'{folder}{fn}.tif', dtype='uint16')
        print(f'>>>> {folder}{fn}.tif saved')
        os.system(f'gdal_translate -of PNG -ot UInt16 -scale {bottom} {top} 0 65535 {folder}{fn}.tif {folder}{fn}.png')
        # colorize
        if vi == 'BSI' or vi == 'T':
            cmap = 'RdYlBu_r'
        else:
            cmap = 'RdYlBu'
        png2color(path=f'{folder}{fn}.png', cmap=cmap, vmin=0, vmax=65535)
        print(f'{folder}{fn}.png SAVED')

        # GET THE METADATA RIGHT
        if i == 0:
            os.system(f'gdalinfo {folder}{fn}.tif > {folder}{fn}.txt')
            m_center, m_ur, m_ll = metadata_gdal_parser(f'{folder}{fn}.txt')

        themeta = {"Metadata":
            {
                "variable": vi,
                "source": satellite,
                "metric_type": metric_type,
                "metric_type2": metric_type2,
                "datetime_range": datetime_range,
                "date": date_,
                "farmgroupid": farmgroupid,
                "center_x": m_center[0],
                "center_y": m_center[1],
                "right": m_ur[0],
                "upper": m_ur[1],
                "left": m_ll[0],
                "lower": m_ll[1],
                "version": version,
                "desc": desc,
                "run": today
            }}

        # UPLOAD to S3 
        layername = f'{satellite} {vi} {metric_type} {metric_type2}'
        filename = f'{folder}{fn}.png'
        objectname = f'{s3folder}{fn}.png'

        if up:
            upload_file(filename,
                        bucketname,
                        objectname)
            add_s3path_to_mapimages(mapimages,
                                    objectname)

    # remove stuff
    if remove_local == True:
        os.system(f'rm -rf {folder}')
        print(f'<<< <<< {folder} removed')

    return {
        "M": {
            "layerName": {
                "S": layername  #This should contain all the information needed to describe this set of images
            },
            "date": {
                "S": datetime_range  #the date range represented in this set of images
            },
            "mapImages": mapimages
        }
    }


def climatology_png_s3(var,
                       farmgroupid: str,
                       bucketname: str,
                       vi: str,
                       satellite: str,
                       metric_type: str,
                       metric_type2: str,
                       bottom: int,
                       top: int,
                       datetime_range: str,
                       version: str,
                       desc: str,
                       run: str,
                       local_folder: str,
                       remove_local=True,
                       up: bool = True,
                       ):
    import os
    from pathlib import Path
    import boto3
    from botocore.exceptions import ClientError
    import logging
    import numpy as np
    import rasterio as rio
    from uuid import uuid4

    mapimages = {"L": []}
    today = str(np.datetime64('today'))
    folder = f'{local_folder}/{farmgroupid}/{today}/{metric_type}/{vi}/{metric_type2}/'
    s3folder = f'{farmgroupid}/{today}/{metric_type}/{vi}/{metric_type2}/'
    Path(folder).mkdir(parents=True, exist_ok=True)

    # upload function
    def upload_file(filename, bucketname, objectname):
        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(filename,
                                             bucketname,
                                             objectname,
                                             ExtraArgs=themeta)

        except ClientError as e:
            logging.error(e)
            return False
        return True

    def add_s3path_to_mapimages(mapImages, path):
        mapImages['L'].append({"M": {'path': {'S': path}}})

    for m in range(0, 12):
        uuid = str(uuid4())
        datestring = str(m + 1).zfill(2)
        fn = f'{datestring}_{uuid}'
        var.isel(month=m).rio.to_raster(f'{folder}{fn}.tif', dtype='uint16')
        print(f'{folder}{fn}.tif SAVED')
        os.system(f'gdal_translate -of PNG -ot UInt16 -scale {bottom} {top} 0 65535 {folder}{fn}.tif {folder}{fn}.png')

        # get the geographic metadata
        if m == 0:
            os.system(f'gdalinfo {folder}{fn}.tif > {folder}{fn}.txt')
            m_center, m_ur, m_ll = metadata_gdal_parser(f'{folder}{fn}.txt')

        # colorize
        if vi == 'BSI' or vi == 'T':
            cmap = 'RdYlBu_r'
        else:
            cmap = 'RdYlBu'
        png2color(path=f'{folder}{fn}.png', cmap=cmap, vmin=0, vmax=65535)
        print(f'{folder}{fn}.png SAVED')

        # Prepare the metadata to be assigned to output images
        themeta = {"Metadata":
            {
                "variable": vi,
                "source": satellite,
                "metric_type": metric_type,
                "metric_type2": metric_type2,
                "datetime_range": datetime_range,
                "date": datestring,
                "farmgroupid": farmgroupid,
                "center_x": m_center[0],
                "center_y": m_center[1],
                "right": m_ur[0],
                "upper": m_ur[1],
                "left": m_ll[0],
                "lower": m_ll[1],
                "version": version,
                "desc": desc,
                "run": today,
            }}

        # UPLOAD to S3 
        layername = f'{satellite} {vi} {metric_type} {metric_type2}'
        filename = f'{folder}{fn}.png'
        objectname = f'{s3folder}{fn}.png'

        if up:
            upload_file(filename,
                        bucketname,
                        objectname)
            add_s3path_to_mapimages(mapimages,
                                    objectname)
    # GET THE RIGHT FOLDER TO REMOVE
    if remove_local == True:
        os.system(f'rm -rf {folder}')

    return {"M": {
        "layerName": {
            "S": layername
        },
        "date": {
            "S": datetime_range
        },
        "mapImages": mapimages
    }
    }


def update_dynamodb(farmgroupid, imageSets):
    dynamodb_client = boto3.client('dynamodb')

    try:
        response = dynamodb_client.update_item(
            TableName='Farm',
            Key={
                'pk': {
                    'S': farmgroupid
                }
            },
            UpdateExpression='set imageSets = :imageSets',
            ExpressionAttributeValues={':imageSets': imageSets},
            ReturnValues="UPDATED_NEW"
        )
        return {
            'statusCode': 200,
            'body': response
        }

    except ClientError as e:
        logging.error(e)
        return False
    return True

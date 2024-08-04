'''
EC2:phenologist downloads ARD 
and process them to calculate indices LAI (kNDVI, EVI, BSI), Phenology and Possibly some stats, save pngs to s3:sanca and index these pngs in the FarmDB.

reads EC2Requests:starterpngqueue list of farms
download ARD from s3:ard4farms
concatenate the series and calculates VIs
  Leaf Area Index (LAI)
    Bare Soil Index (BSI)
    kernel Normalized Difference Vegetation Index (kNDVI)
performs some quality control
 calculates Phenology, 
  and some stats, distribution. 
reescales all data to 1 - 2^16
saves color pngs and uploads to s3:sanca
indexes the lists of images on FarmDB



'''
#%%
import os
import boto3
import xarray as xr
from glob import glob
import rasterio as rio
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt

from dea_tools.temporal import xr_phenology
import numpy as np
import warnings, os
from pprint import pprint
warnings.filterwarnings("ignore")
from utils import open_ncs, treat_save, calculate_indices


vi = 'LAI'
source = f'{vi}_mgap_90_w_7_v2.nc'
mission = 'Landsat'

metric_type2 = 'safra'
years = 0 # all years = 0 or chose year as list of int

if metric_type2 == 'safra':
    start = '06-20'
    end = '03-20'
if metric_type2 == 'safrinha':
    start = '03-20'
    end = '09-20'

# the metrics
basic_pheno_stats = ["SOS","vSOS","POS", "vPOS","EOS", "vEOS",
                     "Trough", "LOS","AOS","ROG","ROS"]
method_sos = "first"
method_eos = "last"

metric_type = 'phenology'
version = '01'
desc = 'LAI based phenology'

remove_temp = False
# more settings
bucketname = 'sanca'
ec2 = False

if ec2:
    up = True
    home = '/home/ubuntu/'
    remove_local = True
else:
    up = True
    home = '/mnt/geodata/Clientes/0FARMS/'
    remove_local = False

bucketnc = 'ard4farms'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucketnc)

dynamodb = boto3.client('dynamodb')
pendingfarms = dynamodb.scan(
                TableName='EC2Requests',
                 )['Items'][1]['pendingfarms']['L']

print(pendingfarms)

#%% AQUI SERA O LOOP
farmgroupid = pendingfarms[2]['S']
print(farmgroupid)

# the routine
dwfolder = f'{home}/{farmgroupid}/' #folder to download ARD 
local_folder = f'{home}/{farmgroupid}/png/'
print(dwfolder,'\n' ,local_folder)

# %% check source
# list contents in the bucket anyways, we will have to grab the date later
contents = [_.key for _ in bucket.objects.all() if farmgroupid in _.key]

if os.path.exists(f'{dwfolder}/results/{source}'):
    print(f'{source} exists. Skipping download and VI calculation')
    vis_f1 = xr.open_dataarray(f'{dwfolder}/results/{source}')
else:
    print(' preparing input data')
    # list files in s3
    
    Path(f'{dwfolder}/').mkdir( parents = True, exist_ok = True)
    # download files if necessary
    for file in contents:
        path = Path(f'{home}/{file}')
        if path.is_file():
            print(f'{file} exists')
        else:
            print(f"{file} doesn't exist")
            os.system(f'aws s3 cp s3://{bucketnc}/{file} {path}')
            print(f'... downloaded')
    # open files
    ds = open_ncs( dwfolder , mission )
    # filtradinha
    for var in ds.data_vars:
        print(f'filtering {var}')
        if var != 'qa_pixel' and var !='blue' and var != 'T' :
            ds[var] = xr.where((ds[var] > 30000) | (ds[var] < 5000), np.nan, ds[var])
        if var == 'blue':
            ds[var] = xr.where((ds[var] > 15000) | (ds[var] < 5000), np.nan, ds[var]) 
    # resample
    ds1 = ds.resample(time='W').max(skipna=True)
    print(ds.dims)
    del ds
    print(ds1.dims)
    print(f'\n ! source files for {vi} loaded and filtered \n')
    # calculate VI - LAI
    vis = calculate_indices( ds1 , index = [vi], drop = True, collection = mission)
    vis[vi] = xr.where((vis[vi] > 7.4) | (vis[vi] < 0), np.nan, vis[vi])
    # treat and save
    vis_f1 = treat_save(vis = vis, vi = vi, days = 90, window = 7, folderout = f'{dwfolder}/results/', save = True)
print(vis_f1.sizes)

# %% some checks
nextyear = 0
if years == 0:
    print('o-o check yeaars \n')
    if int(end[:2]) > int(start[:2]):
        print('o-o end > start: season start and finish in the same year \n    typically safrinha or northern hemisphere')#and int(end[:2]) < int(str(df.time.values[-1]).split('T')[0][5:7]):
        if int(end[:2]) > int(str(vis_f1.time.values[-1]).split('T')[0][5:7]):
            print('!!! not enough data for the last year \n   end > data.end')
            years = np.unique(vis_f1.time.dt.year)[:-1]
        else:
            print('! you are good to go')
            years = np.unique(vis_f1.time.dt.year)
            print('> selected years:', years)
    else: 
        print('o-o you are looking at the tropics and southern hemisphere, \n    since your season go to the next year (+1)')
        nextyear = 1
        years = np.unique(vis_f1.time.dt.year)[:-1]
        print('> selected years:', years, ' (+1)')
else:
    print('o-o your customized choice')
    years = sorted(years)
    if int(end[:2]) < int(start[:2]):
        print(f' end < start : {end[:2]} < {start[:2]}')
        nextyear = 1
        print('> selected years: ', years, ' (+1)')
    else:
        print('> selected years:', years)

# %% run phenology calculation
dates = []
for year in years:
    da = vis_f1.sel(time=slice(f'{year}-{start}',f'{year+nextyear}-{end}'))
    da = da.compute()
    metrics = xr_phenology( da, method_sos=method_sos, method_eos=method_eos, stats=basic_pheno_stats,
                    verbose=False )
    # add results to dict
    #datelabel = f'{year}-06-20'
    date = np.datetime64(f'{year}-{end}')
    dates.append(date)
    metrics = metrics.expand_dims(dim='time')
    # save temoporary files
    metrics.to_netcdf(f'{dwfolder}/results/phenology_{year}_{metric_type2}.nc')
    print(f'{dwfolder}/results/phenology_{year}_{metric_type2}.nc')
del da, metrics
# %% 
files = sorted(glob(f'{dwfolder}/results/phenology_*_{metric_type2}.nc'))
pprint(files)

# %%
DF = xr.open_mfdataset(files, concat_dim='time', combine='nested')
DF = DF.assign_coords({'time':dates})
datei, datef = str(DF.time.values[0])[:4] , str(DF.time.values[-1])[:4]
DF.to_netcdf(f'{dwfolder}/results/Phenology_{datei}-{datef}_{metric_type2}.nc')
if remove_temp == True:
    os.system(f'rm {dwfolder}/results/phe*')
    print('!! temporary phenology* files removed')
print(f'o-o ! saved >> {dwfolder}/results/Phenology_{datei}-{datef}_{metric_type2}.nc')

# APPLY HARD LIMITS based on LAI
L = {'ROG' : (-0.11 , 0.11), 
    'ROS' : (-0.1 , 0.1), 
    'AOS' : (0.1 , 8),
    'Trough' : (0.1 , 8),
    'vSOS' : (0.1 , 8),
    'vPOS' : (0.1 , 8),
    'vEOS' : (0.1 , 8),
    'SOS' : (1, 365),
    'POS' : (1, 365),
    'EOS' : (1, 365),
    'LOS' : (1, 365)
        }

df2 = DF.copy()
for metric in list(df2):
    df2[metric] = xr.where( ( df2[metric].astype('float32') < L[metric][0]) | 
                            ( df2[metric].astype('float32') > L[metric][1]), 
                              np.nan, df2[metric].astype('float32'))

bottom, top = 1 , 10000

# %% INTERPOLAO USANDO HARD LIMITS # TALVEZ SEJA O CASO DE .clip(min,max)
for metric in list(df2):
    df2[metric].values = np.interp(df2[metric].values, (L[metric][0], L[metric][1]), (bottom, top))
    print(metric, np.nanmin(df2[metric].values), np.nanmax(df2[metric].values))
    df2[metric] = df2[metric].astype('uint16')

# %%
imageSets = {"L": []}
def appendImageSet(imageSet): 
    imageSets['L'].append(imageSet)

today = str(np.datetime64('today'))

appendImageSet(phenology_png_s3(var = df2, dates = dates, farmgroupid = farmgroupid, bucketname = bucketname, 
                                vi = vi, mission = mission, metric_type = metric_type,
                                metric_type2=metric_type2, bottom = bottom, top = top, version = version,
                                desc = desc, run = today, local_folder=local_folder,
                                remove_local=False, up=up))

print(f'updating dynamodb Farm table for {farmgroupid}')
update_dynamodb(farmgroupid, imageSets)

# remove item from table
# response = dynamodb.update_item(
# TableName='EC2Requests',
# Key = {
#     'pk':{
#         'S': 'starterpngqueue'
#     }
# },
# UpdateExpression='remove pendingfarms[0]',
# )
# print(f' {farmgroupid} processed and removed from starterpngqueue.')

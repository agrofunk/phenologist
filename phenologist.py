'''
    Introducing the Phenologist

    enter 
    -> a farmgroupid from a QUEUE
    -> a source for Vegetation Index time-series file
    -> suffix: a name like safra, safrinha
    -> month-day start, e.g.: '06-20' or '03-20'
    -> month-day end, e.g.: '04-20' or 09-20

'''

# %%
from timeit import default_timer as timer
t_start = timer()

import xarray as xr
from dea_tools.temporal import xr_phenology, temporal_statistics
import numpy as np
import warnings, os
from pprint import pprint
warnings.filterwarnings("ignore")
from pathlib import Path
from glob import glob


import pylab as plt


# %% payload
farmgroupid = 'fd9239de-65ae-4d0f-a1cf-9c0fd5ac9131'
vi = 'LAI'
source = f'{vi}_mgap_90_w_7_v2.nc'

suffix = 'safrinha'
years = 0 # all years = 0 or chose year as list of int

if suffix == 'safra':
    start = '06-20'
    end = '03-20'
if suffix == 'safrinha':
    start = '03-20'
    end = '09-20'

# the metrics
basic_pheno_stats = ["SOS","vSOS","POS", "vPOS","EOS", "vEOS",
                     "Trough", "LOS","AOS","ROG","ROS"]
method_sos = "first"
method_eos = "last"

remove_temp = True

# %% Load data and prepare output folder
# Open VI file
file = f'/mnt/geodata/Clientes/0FARMS/{farmgroupid}/results/{source}'
print(f'source file for {vi} is {file}')
df = xr.open_dataset(file)
folderout = f'/mnt/geodata/Clientes/0FARMS/{farmgroupid}/results/Phenology/'
Path(f'{folderout}').mkdir( parents = True, exist_ok = True)

# %% some checks
nextyear = 0
if years == 0:
    print('o-o check yeaars \n')
    if int(end[:2]) > int(start[:2]):
        print('o-o end > start: season start and finish in the same year \n    typically safrinha or northern hemisphere')#and int(end[:2]) < int(str(df.time.values[-1]).split('T')[0][5:7]):
        if int(end[:2]) > int(str(df.time.values[-1]).split('T')[0][5:7]):
            print('!!! not enough data for the last year \n   end > data.end')
            years = np.unique(df.time.dt.year)[:-1]
        else:
            print('! you are good to go')
            years = np.unique(df.time.dt.year)
            print('> selected years:', years)
    else: 
        print('o-o you are looking at the tropics and southern hemisphere, \n    since your season go to the next year (+1)')
        nextyear = 1
        years = np.unique(df.time.dt.year)[:-1]
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
    da = df['LAI'].sel(time=slice(f'{year}-{start}',f'{year+nextyear}-{end}'))
    da = da.compute()
    metrics = xr_phenology( da, method_sos=method_sos, method_eos=method_eos, stats=basic_pheno_stats,
                    verbose=False )
    # add results to dict
    #datelabel = f'{year}-06-20'
    date = np.datetime64(f'{year}-{end}')
    dates.append(date)
    metrics = metrics.expand_dims(dim='time')
    # save temoporary files
    metrics.to_netcdf(f'{folderout}/phenology_{year}_{suffix}.nc')
    del da, metrics


files = sorted(glob(f'{folderout}/phenology*{suffix}.nc'))
pprint(files)

DF = xr.open_mfdataset(files, concat_dim='time', combine='nested')
DF = DF.assign_coords({'time':dates})
datei, datef = str(DF.time.values[0])[:4] , str(DF.time.values[-1])[:4]
DF.to_netcdf(f'{folderout}/Phenology_{datei}-{datef}_{suffix}.nc')
if remove_temp == True:
    os.system(f'rm {folderout}/phe*')
    print('!! temporary phenology* files removed')
print(f'o-o ! saved >> {folderout}/Phenology_{datei}-{datef}_{suffix}.nc')
t_end = timer()
print('\n total execution time: \n ',t_end - t_start, ' seconds')

# %% FIXME TENTEI OUTRAS STATS, SO DAH NAN
# stats = [
#     "f_mean",
#     "median_change",
#     "abs_change",
#     "complexity",
#     "central_diff",
#     "discordance",
# ]
# df = df.rename({'latitude' : 'y' , 'longitude': 'x'})
# ts_stats = temporal_statistics(df['LAI'], stats)
# print(ts_stats)

# # %% um PLOT
# #ts_stats = ts_stats.where(cm == 1)

# # set up figure
# fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True, sharex=True )

# # set aspect ratios
# for a in fig.axes:
#     a.set_aspect("equal")

# # set colorbar size
# cbar_size = 0.7
# stats = list(ts_stats.data_vars)

# # plot
# axx = 0
# col = 0
# for st in stats:
#     if axx > 3:
#         col += 1
#         axx = 0
#     ts_stats[st].plot(ax=axes[col][axx], cmap="plasma",cbar_kwargs=dict(shrink=cbar_size, label=None))
#     axes[col][axx].set_title(st)
#     axx += 1
    
# if len(stats) < 8: fig.delaxes(axes[1][3])

# plt.tight_layout();
# # %%

# %%

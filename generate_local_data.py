# This script was adapted from Jacob Tomlinson's 1BRC submission
# https://github.com/gunnarmorling/1brc/discussions/487
# Edited by David Goldstrom
# Last Edited Date: 25 Nov 2025

# %% Imports

import os
import shutil
import numpy as np
import pandas as pd

from dask.distributed import LocalCluster, progress
import multiprocessing as mp

# %% Set Variables
n = 1_000_000_000  # Total number of rows of data to generate
chunksize = 10_000_000  # Number of rows of data per file
std = 10.0  # Assume normally distributed temperatures with a standard deviation of 10

os.chdir("C:\\Temp") # set a working directory on the local drive due to file size generated
# print("Current Working Directory:", os.getcwd())

# Lookup table of stations and their mean temperatures
# this is a small file to use to generate big data file
lookup_df = \
pd.read_csv("C:\\Users\\VHAATGGoldsD15\\OneDrive - Department of Veterans Affairs\\Documents\\_Python\\1trc\\lookup.csv")  


PARQUET_DIR = "bigData"
# if os.path.exists(PARQUET_DIR):
#     shutil.rmtree(PARQUET_DIR)
os.makedirs(PARQUET_DIR, exist_ok=True)
print(f"Parquet directory set to: {PARQUET_DIR}")

# %% Function Definition to Generate Data
def generate_chunk(partition_idx, chunksize, std, lookup_df):
    """Generate some sample data based on the lookup table."""
    
    rng = np.random.default_rng(partition_idx)  # Deterministic data generation
    df = pd.DataFrame(
        {
            # Choose a random station from the lookup table for each row in our output
            "station": rng.integers(0, len(lookup_df) - 1, int(chunksize)),
            # Generate a normal distibution around zero for each row in our output
            # Because the std is the same for every station we can adjust the mean for each row afterwards
            "measure": rng.normal(0, std, int(chunksize)),
        }
    )

    # Offset each measurement by the station's mean value
    df.measure += df.station.map(lookup_df.mean_temp)
    # Round the temprature to one decimal place
    df.measure = df.measure.round(decimals=1)
    # Convert the station index to the station name
    df.station = df.station.map(lookup_df.station)

    # Save this chunk to the output file
    filename = f"measurements-{partition_idx}.parquet"
    local = os.path.join(PARQUET_DIR, filename)
    df.to_parquet(local, engine="pyarrow")
    

# %% Main Program

# Generate partitioned dataset
cluster = LocalCluster(
    n_workers=int(0.9 * mp.cpu_count()), # 90% of CPU cores
    processes=True, # Use processes instead of threads
    threads_per_worker=1, # One thread per worker
    memory_limit='2GB', # Memory limit per worker
    #n_workers=4, threads_per_worker=2
)
client = cluster.get_client()  # set up local cluster
client

# Generate partitioned dataset
results = client.map(
    generate_chunk,
    range(int(n / chunksize)),
    chunksize=chunksize,
    std=std,
    lookup_df=lookup_df,
)
progress(results) # this computes the delayed results and monitors to completion

# close local cluster
client.close()
cluster.close()





"""
Name
----
WFC3 Module

Purpose
-------
This module contains helper functions for downloading, extracting, and plotting
WFC3 data.

Use
---
This module is intended to be imported in a Jupyter notebook:

    >>> import wfc3_module

Author
------
Fred Dauphin, June 2023

"""

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import median_abs_deviation

from astroquery.mast import Observations
from astropy.io import fits
from ginga.util.zscale import zscale

def get_obs_from_query_criteria(df, obs_id, proposal_id):
    ''' 
    Get observations from Astroquery using an observation ID,
    or a Proposal ID.
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with observation metadata from WFC3 QL.
    obs_id : str
        Observation ID.
    proposal_id : str
        Proposal ID.
        
    Returns
    -------
    obs : astropy.Table
        Observations to be downloaded.
    '''
    
    # Get observations using observation ID
    obs = Observations.query_criteria(obs_id=obs_id)
    df_rootname = df[df['rootname'] == obs_id]
    
    # Return if observations can be matched
    if len(obs) > 0:
        return obs
    
    # If observations cannot be matched, try using proposal ID
    elif proposal_id:
        obs = Observations.query_criteria(proposal_id=proposal_id)
    
    # If observations still cannot be matched, try finding a proposal ID
    elif len(obs) == 0:
        try:
            proposal_id = df_rootname['proposid']
        except KeyError:
            msg = f'proposal_id not found in the dataframe. \
                    Look for Proposal ID on QL for {obs_id}.'
            raise KeyError(msg)
    
    # Find observation ID candidates for a corresponding filter
    filt = df_rootname['filter'].iloc[0]
    obs_filt = obs[obs['filters'] == filt]
    obs_id_cands = obs_filt['obs_id'].value.data
    
    # Loop through the matched observations and 
    # find the appropriate association ID
    for obs_id_cand in obs_id_cands:
        if obs_id[:6] == obs_id_cand[:6]:
            asn_id = obs_id_cand
    
    # Use association ID for observations
    obs = obs[obs['obs_id'] == asn_id]
    
    return obs

def get_obs_from_query_criteria_wildcard(obs_id, filt):
    ''' 
    Get observations from Astroquery using an observation ID wildcard.
    
    Parameters
    ----------
    obs_id : str
        Observation ID.
    filt : str
        Filter used.
        
    Returns
    -------
    obs : astropy.Table
        Observations to be downloaded.
    '''
    
    asn_id_wildcard = obs_id[:6] + '*'
    obs = Observations.query_criteria(obs_id=asn_id_wildcard, filters=filt)
    
    return obs

def download_raw_flt_fits(obs, obs_id):
    '''
    Download RAW and FLT fits files of an observation.
    
    Parameters
    ----------
    obs : atropy.Table
        Observations to be downloaded.
    obs_id : str
        Observation ID.
    '''
    
    # Get product list and download files
    ext = ['_raw.fits', '_flt.fits']
    prods = Observations.get_product_list(obs)
    prods_filtered = Observations.filter_products(prods, 
                                                  obs_id=[obs_id], 
                                                  extension=ext)
    Observations.download_products(prods_filtered, mrp_only=False, cache=False)
    
def get_raw_flt_data(obs_id):
    '''
    Extract RAW and FLT data from downloaded fits file.
    
    For RAW, we only open the SCI array because it does not have an 
    ERR or DQ array. For FLT, we open the SCI, ERR, and DQ arrays.
    Since UVIS has two chips, we check if UVIS images uses both chips
    to indicate whether or not there is a second set of arrays to be opened.
    
    Parameters
    ----------
    obs_id : str
        Observation ID.
    
    Returns
    -------
    data : list
        The RAW SCI, FLT SCI, FLT ERR, and FLT DQ arrays.
    '''
    
    # Find data in directories
    paths_fits = sorted(glob(f'mastDownload/HST/{obs_id}/*'))
    file_flt, file_raw = paths_fits
    
    # Open data
    data_raw = fits.getdata(file_raw, 'SCI', 1)
    data_sci = fits.getdata(file_flt, 'SCI', 1)
    data_err = fits.getdata(file_flt, 'ERR', 1)
    data_dq = fits.getdata(file_flt, 'DQ', 1)
    
    # If data uses both UVIS chips, open the second set of arrays
    detector = fits.getval(file_flt, 'detector')
    subarray = fits.getval(file_flt, 'subarray')
    
    if detector == 'UVIS' and subarray == False:
        data_raw2 = fits.getdata(file_raw, 'SCI', 2)
        data_sci2 = fits.getdata(file_flt, 'SCI', 2)
        data_err2 = fits.getdata(file_flt, 'ERR', 2)
        data_dq2 = fits.getdata(file_flt, 'DQ', 2)
        
        data_raw = np.vstack((data_raw, data_raw2))
        data_sci = np.vstack((data_sci, data_sci2))
        data_err = np.vstack((data_err, data_err2))
        data_dq = np.vstack((data_dq, data_dq2))

    data = [data_raw, data_sci, data_err, data_dq]

    return data

def plot_images(obs_id, data, dq_flag):
    '''
    Plot the RAW SCI, FLT SCI, FLT ERR, and one flag from the FLT DQ.
    
    Parameters
    ----------
    obs_id : str:
        Observation ID.
    data : list
        The RAW SCI, FLT SCI, FLT ERR, and FLT DQ arrays.
    dq_flag : int
        The integer bit associated with a data quality flag.
        Possible values are the powers of 2 from 0 to 15.
    
    '''
    
    subplot_titles = ['RAW (DN)', 'FLT SCI (e-)', 'FLT ERR (e-)', 'FLT DQ']
    
    # Loop through to plot RAW, FLT SCI, and FLT ERR
    for i in range (3):
        vmin, vmax = zscale(data[i])
        plt.title(f'{obs_id} {subplot_titles[i]}')
        plt.imshow(data[i], vmin=vmin, vmax=vmax, origin='lower', cmap='gray')
        plt.colorbar()
        plt.show()
        
    # If flag is 0, explicitly use boolean logic
    if dq_flag == 0:
        data_dq_one_flag = data[3] == 0
    else:
        data_dq_one_flag = np.bitwise_and(data[3], dq_flag)
    
    # Plot DQ
    plt.figure(figsize=[5,5])
    plt.title(f'{obs_id} {subplot_titles[3]}: Flag == {dq_flag}')
    plt.imshow(data_dq_one_flag, origin='lower', cmap='gray')
    plt.tick_params()
    plt.show()
    
def plot_histograms(obs_id, data_sci):
    '''
    Plot log scale histograms of the FLT SCI array (linear and log scale).
    
    Parameters
    ----------
    obs_id : str:
        Observation ID.
    data_sci : array-like
        The FLT SCI array.

    '''
    
    fontsize = 15
    fig, axs = plt.subplots(1,2,figsize=[15,5])
    
    # Plot histogram (linear scale)
    axs[0].set_title(f'{obs_id} Histogram', fontsize=fontsize)
    axs[0].hist(data_sci.flatten(), bins=100)
    axs[0].set_xlabel('e-', fontsize=fontsize)
    axs[0].set_ylabel('Frequency', fontsize=fontsize)
    axs[0].set_yscale('log')
    axs[0].tick_params(labelsize=fontsize)
    
    # Plot histogram (log scale)
    data_log = np.log10(data_sci[data_sci>0].flatten())
    axs[1].set_title(f'{obs_id} Histogram (Data > 0; Log scale)', 
                     fontsize=fontsize)
    axs[1].hist(data_log, bins=100)
    axs[1].set_xlabel('log(e-)', fontsize=fontsize)
    axs[1].set_ylabel('Frequency', fontsize=fontsize)
    axs[1].set_yscale('log')
    axs[1].tick_params(labelsize=fontsize)
    
def plot_sci_lt_0(obs_id, data_sci, y_min, x_min, size):
    '''
    Plot the FLT SCI array showcasing where pixels are negative,
    plot a subsection of that array, and print the number of negative pixels.
    
    Parameters
    ----------
    obs_id : str:
        Observation ID.
    data_sci : array-like
        The FLT SCI array.
    y_min : int
        The minimum y pixel for the subsection.
    x_min : int
        The minimum x pixel for the subsection.
    size : int
        The size of the subsection.

    '''
    
    data_sci_lt_0 = data_sci <= 0
    fig, axs = plt.subplots(1,2,figsize=[10,5])
    
    # Plot image
    axs[0].set_title(f'{obs_id} FLT SCI:\n <= 0')
    axs[0].imshow(data_sci_lt_0, origin='lower')

    # Define variables for subsection
    y_max = y_min + size
    x_max = x_min + size

    # Plot subsection
    subtitle = f'y, x, size = {y}, {x}, {size}'
    axs[1].set_title(f'{obs_id} FLT SCI:\n <= 0 (subsection)\n {subtitle}')
    axs[1].imshow(data_sci_lt_0[y_min:y_max, x_min:x_max], origin='lower')
    
    # Print number of pixels below 0
    percent = 100 * data_sci_lt_0.sum() / (data_sci.shape[0] * data_sci.shape[1])
    print (f'{percent:.3f}% of pixels are less than 0')
    
def get_stats(obs_id, data_sci):
    '''
    Calculate the following statistics for the FLT SCI array:
    mean, median, min, max, standard deviation, and median absolute deviation.
    
    Parameters
    ----------
    obs_id : str:
        Observation ID.
    data_sci : array-like
        The FLT SCI array.

    Returns
    ------
    data_sci_stats = pandas.DataFrame
        The summary statistics of the FLT SCI array.
    '''
    
    # Calculate stats
    data_sci_mean = np.mean(data_sci)
    data_sci_median = np.median(data_sci)
    data_sci_min = np.min(data_sci)
    data_sci_max = np.max(data_sci)
    data_sci_std = np.std(data_sci)
    data_sci_mad = median_abs_deviation(data_sci.flatten())
    
    # Organize stats into a DataFrame
    data_sci_stats = [obs_id, data_sci_mean, data_sci_median, 
                      data_sci_min, data_sci_max, 
                      data_sci_std, data_sci_mad]
    
    columns = ['obs_id', 'mean', 'median', 'min', 'max', 'std', 'mad']
    data_sci_stats = pd.DataFrame([data_sci_stats], columns=columns)
    
    return data_sci_stats

if __name__ == '__main__':
    
    print ('I am now a script')
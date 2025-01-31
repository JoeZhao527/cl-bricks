import pandas as pd
from tqdm import tqdm
import numpy as np
from zipfile import ZipFile
import pickle
from scipy.stats import linregress
from scipy.interpolate import interp1d
import tsfel
import os
import json
import warnings
import concurrent.futures
from ensemble.config.paths import PATHS
import datetime
import argparse
warnings.filterwarnings("ignore")
np.random.seed(83355)

# Initialize argparse
parser = argparse.ArgumentParser(description="Process and shuffle chunks for time series data")
parser.add_argument('--train_zip_path', type=str, required=True, help="Path to the train zip file")
parser.add_argument('--n_splits', type=int, nargs='+', default=[1, 2], help="List of chunk sizes to shuffle")
parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save the results")

# Parse arguments
args = parser.parse_args()

def log(msg):
    print(f"[{datetime.datetime.now()}] Main Log: {msg}")

log(f"Get train signal zip path: {args.train_zip_path}")
log(f"Get n_splits: {args.n_splits} {type(args.n_splits)}")
log(f"Get output_dir: {args.output_dir}")

folder_name = os.path.join(args.output_dir, 'data_features')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
folder_name2 = os.path.join(args.output_dir, 'data_features_fix')
if not os.path.exists(folder_name2):
    os.makedirs(folder_name2)

dt = 4838397.067/85922 # from max val and largest recording

train_y = pd.read_csv(PATHS.train_y_path)
zipf = ZipFile(args.train_zip_path, 'r')
zipftest = ZipFile(PATHS.test_zip_path, 'r')
listtestfile = zipftest.namelist()[1:]

columnlist = ['Active_Power_Sensor', 'Air_Flow_Sensor', 'Air_Flow_Setpoint', 'Air_Temperature_Sensor', 'Air_Temperature_Setpoint', 'Alarm', 'Angle_Sensor', 'Average_Zone_Air_Temperature_Sensor', 'Chilled_Water_Differential_Temperature_Sensor', 'Chilled_Water_Return_Temperature_Sensor', 'Chilled_Water_Supply_Flow_Sensor', 'Chilled_Water_Supply_Temperature_Sensor', 'Command', 'Cooling_Demand_Sensor', 'Cooling_Demand_Setpoint', 'Cooling_Supply_Air_Temperature_Deadband_Setpoint', 'Cooling_Temperature_Setpoint', 'Current_Sensor', 'Damper_Position_Sensor', 'Damper_Position_Setpoint', 'Demand_Sensor', 'Dew_Point_Setpoint', 'Differential_Pressure_Sensor', 'Differential_Pressure_Setpoint', 'Differential_Supply_Return_Water_Temperature_Sensor', 'Discharge_Air_Dewpoint_Sensor', 'Discharge_Air_Temperature_Sensor', 'Discharge_Air_Temperature_Setpoint', 'Discharge_Water_Temperature_Sensor', 'Duration_Sensor', 'Electrical_Power_Sensor', 'Energy_Usage_Sensor', 'Filter_Differential_Pressure_Sensor', 'Flow_Sensor', 'Flow_Setpoint', 'Frequency_Sensor', 'Heating_Demand_Sensor', 'Heating_Demand_Setpoint', 'Heating_Supply_Air_Temperature_Deadband_Setpoint', 'Heating_Temperature_Setpoint', 'Hot_Water_Flow_Sensor', 'Hot_Water_Return_Temperature_Sensor', 'Hot_Water_Supply_Temperature_Sensor', 'Humidity_Setpoint', 'Load_Current_Sensor', 'Low_Outside_Air_Temperature_Enable_Setpoint', 'Max_Air_Temperature_Setpoint', 'Min_Air_Temperature_Setpoint', 'Outside_Air_CO2_Sensor', 'Outside_Air_Enthalpy_Sensor', 'Outside_Air_Humidity_Sensor', 'Outside_Air_Lockout_Temperature_Setpoint', 'Outside_Air_Temperature_Sensor', 'Outside_Air_Temperature_Setpoint', 'Parameter', 'Peak_Power_Demand_Sensor', 'Position_Sensor', 'Power_Sensor', 'Pressure_Sensor', 'Rain_Sensor', 'Reactive_Power_Sensor', 'Reset_Setpoint', 'Return_Air_Temperature_Sensor', 'Return_Water_Temperature_Sensor', 'Room_Air_Temperature_Setpoint', 'Sensor', 'Setpoint', 'Solar_Radiance_Sensor', 'Speed_Setpoint', 'Static_Pressure_Sensor', 'Static_Pressure_Setpoint', 'Status', 'Supply_Air_Humidity_Sensor', 'Supply_Air_Static_Pressure_Sensor', 'Supply_Air_Static_Pressure_Setpoint', 'Supply_Air_Temperature_Sensor', 'Supply_Air_Temperature_Setpoint', 'Temperature_Sensor', 'Temperature_Setpoint', 'Thermal_Power_Sensor', 'Time_Setpoint', 'Usage_Sensor', 'Valve_Position_Sensor', 'Voltage_Sensor', 'Warmest_Zone_Air_Temperature_Sensor', 'Water_Flow_Sensor', 'Water_Temperature_Sensor', 'Water_Temperature_Setpoint', 'Wind_Direction_Sensor', 'Wind_Speed_Sensor', 'Zone_Air_Dewpoint_Sensor', 'Zone_Air_Humidity_Sensor', 'Zone_Air_Humidity_Setpoint', 'Zone_Air_Temperature_Sensor' ]

cfg_file1 = tsfel.get_features_by_domain(['statistical', 'temporal'])
       
jsondata= """{
  "spectral": {
    "Spectrogram mean coefficient": {
      "complexity": "constant",
      "description": "Calculates the average value for each frequency in the spectrogram over the entire duration of the signal.",
      "function": "tsfel.spectrogram_mean_coeff",
      "parameters": {
        "fs": 100,
        "bins": 32
      },
      "n_features": "bins",
      "use": "yes"
    },
    "Fundamental frequency": {
      "complexity": "log",
      "description": "Computes the fundamental frequency.",
      "function": "tsfel.fundamental_frequency",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Human range energy": {
      "complexity": "log",
      "description": "Computes the human range energy ratio given by the ratio between the energy in frequency 0.6-2.5Hz and the whole energy band.",
      "function": "tsfel.human_range_energy",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes",
      "tag": "inertial"
    },
    "Max power spectrum": {
      "complexity": "log",
      "description": "Computes the maximum power spectrum density.",
      "function": "tsfel.max_power_spectrum",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Maximum frequency": {
      "complexity": "log",
      "description": "Computes the maximum frequency.",
      "function": "tsfel.max_frequency",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Median frequency": {
      "complexity": "log",
      "description": "Computes the median frequency.",
      "function": "tsfel.median_frequency",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Power bandwidth": {
      "complexity": "log",
      "description": "Computes power spectrum density bandwidth of the signal.",
      "function": "tsfel.power_bandwidth",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral centroid": {
      "complexity": "linear",
      "description": "Computes the barycenter of the spectrum.",
      "function": "tsfel.spectral_centroid",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes",
      "tag": "audio"
    },
    "Spectral decrease": {
      "complexity": "log",
      "description": "Computes the amount of decreasing of the spectra amplitude.",
      "function": "tsfel.spectral_decrease",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral distance": {
      "complexity": "log",
      "description": "Computes the signal spectral distance.",
      "function": "tsfel.spectral_distance",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral entropy": {
      "complexity": "log",
      "description": "Computes the spectral entropy of the signal based on Fourier transform.",
      "function": "tsfel.spectral_entropy",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes",
      "tag": "eeg"
    },
    "Spectral kurtosis": {
      "complexity": "linear",
      "description": "Computes the flatness of a distribution around its mean value.",
      "function": "tsfel.spectral_kurtosis",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral positive turning points": {
      "complexity": "log",
      "description": "Computes number of positive turning points of the fft magnitude signal",
      "function": "tsfel.spectral_positive_turning",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral roll-off": {
      "complexity": "log",
      "description": "Computes the frequency where 95% of the signal magnitude is contained below of this value.",
      "function": "tsfel.spectral_roll_off",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes",
      "tag": "audio"
    },
    "Spectral roll-on": {
      "complexity": "log",
      "description": "Computes the frequency where 5% of the signal magnitude is contained below of this value.",
      "function": "tsfel.spectral_roll_on",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral skewness": {
      "complexity": "linear",
      "description": "Computes the asymmetry of a distribution around its mean value.",
      "function": "tsfel.spectral_skewness",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral slope": {
      "complexity": "log",
      "description": "Computes the spectral slope, obtained by linear regression of the spectral amplitude.",
      "function": "tsfel.spectral_slope",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral spread": {
      "complexity": "linear",
      "description": "Computes the spread of the spectrum around its mean value.",
      "function": "tsfel.spectral_spread",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Spectral variation": {
      "complexity": "log",
      "description": "Computes the amount of variation of the spectrum along time.",
      "function": "tsfel.spectral_variation",
      "parameters": {
        "fs": 100
      },
      "n_features": 1,
      "use": "yes"
    },
    "Wavelet absolute mean": {
      "complexity": "linear",
      "description": "Computes CWT absolute mean value of each wavelet scale.",
      "function": "tsfel.wavelet_abs_mean",
      "parameters": {
          "fs": 100,
          "max_width": 10,
          "wavelet": "mexh"
      },
      "tag": [
          "eeg",
          "ecg"
      ],
      "n_features": "max_width",
      "use": "yes"
    },
    "Wavelet energy": {
        "complexity": "linear",
        "description": "Computes CWT energy of each wavelet scale.",
        "function": "tsfel.wavelet_energy",
        "parameters": {
            "fs": 100,
            "max_width": 10,
            "wavelet": "mexh"
        },
        "tag": "eeg",
        "n_features": "max_width",
        "use": "yes"
    },
    "Wavelet entropy": {
        "complexity": "linear",
        "description": "Computes CWT entropy of the signal.",
        "function": "tsfel.wavelet_entropy",
        "parameters": {
            "fs": 100,
            "max_width": 10,
            "wavelet": "mexh"
        },
        "tag": "eeg",
        "n_features": "max_width",
        "use": "yes"
    },
    "Wavelet standard deviation": {
        "complexity": "linear",
        "description": "Computes CWT std value of each wavelet scale.",
        "function": "tsfel.wavelet_std",
        "parameters": {
            "fs": 100,
            "max_width": 10,
            "wavelet": "mexh"
        },
        "n_features": "max_width",
        "use": "yes"
    },
    "Wavelet variance": {
        "complexity": "linear",
        "description": "Computes CWT variance value of each wavelet scale.",
        "function": "tsfel.wavelet_var",
        "parameters": {
            "fs": 100,
            "max_width": 10,
            "wavelet": "mexh"
        },
        "tag": "eeg",
        "n_features": "max_width",
        "use": "yes"
    }
    
    }
    }"""
    
cfg_file2 = json.loads(jsondata)
   
columnlist_test = ['0_Absolute energy',  '0_Area under the curve',  '0_Autocorrelation',  '0_Average power',  '0_Centroid',  '0_ECDF Percentile Count_0',  '0_ECDF Percentile Count_1',  '0_ECDF Percentile_0',  '0_ECDF Percentile_1',  '0_ECDF_0',  '0_ECDF_1',  '0_ECDF_2',  '0_ECDF_3',  '0_ECDF_4',  '0_ECDF_5',  '0_ECDF_6',  '0_ECDF_7',  '0_ECDF_8',  '0_ECDF_9',  '0_Entropy',  '0_Histogram mode',  '0_Interquartile range',  '0_Kurtosis',  '0_Max',  '0_Mean',  '0_Mean absolute deviation',  '0_Mean absolute diff',  '0_Mean diff',  '0_Median',  '0_Median absolute deviation',  '0_Median absolute diff',  '0_Median diff',  '0_Min',  '0_Negative turning points',  '0_Neighbourhood peaks',  '0_Peak to peak distance',  '0_Positive turning points',  '0_Root mean square',  '0_Signal distance',  '0_Skewness',  '0_Slope',  '0_Standard deviation',  '0_Sum absolute diff',  '0_Variance',  '0_Zero crossing rate',  '0_Fundamental frequency',  '0_Human range energy',  '0_Max power spectrum',  '0_Maximum frequency',  '0_Median frequency',  '0_Power bandwidth',  '0_Spectral centroid',  '0_Spectral decrease',  '0_Spectral distance',  '0_Spectral entropy',  '0_Spectral kurtosis',  '0_Spectral positive turning points',  '0_Spectral roll-off',  '0_Spectral roll-on',  '0_Spectral skewness',  '0_Spectral slope',  '0_Spectral spread',  '0_Spectral variation',  '0_Spectrogram mean coefficient_0.0Hz',  '0_Spectrogram mean coefficient_1.03Hz',  '0_Spectrogram mean coefficient_10.31Hz',  '0_Spectrogram mean coefficient_11.34Hz',  '0_Spectrogram mean coefficient_12.37Hz',  '0_Spectrogram mean coefficient_13.4Hz',  '0_Spectrogram mean coefficient_14.44Hz',  '0_Spectrogram mean coefficient_15.47Hz',  '0_Spectrogram mean coefficient_16.5Hz',  '0_Spectrogram mean coefficient_17.53Hz',  '0_Spectrogram mean coefficient_18.56Hz',  '0_Spectrogram mean coefficient_19.59Hz',  '0_Spectrogram mean coefficient_2.06Hz',  '0_Spectrogram mean coefficient_20.62Hz',  '0_Spectrogram mean coefficient_21.65Hz',  '0_Spectrogram mean coefficient_22.68Hz',  '0_Spectrogram mean coefficient_23.72Hz',  '0_Spectrogram mean coefficient_24.75Hz',  '0_Spectrogram mean coefficient_25.78Hz',  '0_Spectrogram mean coefficient_26.81Hz',  '0_Spectrogram mean coefficient_27.84Hz',  '0_Spectrogram mean coefficient_28.87Hz',  '0_Spectrogram mean coefficient_29.9Hz',  '0_Spectrogram mean coefficient_3.09Hz',  '0_Spectrogram mean coefficient_30.93Hz',  '0_Spectrogram mean coefficient_31.97Hz',  '0_Spectrogram mean coefficient_4.12Hz',  '0_Spectrogram mean coefficient_5.16Hz',  '0_Spectrogram mean coefficient_6.19Hz',  '0_Spectrogram mean coefficient_7.22Hz',  '0_Spectrogram mean coefficient_8.25Hz',  '0_Spectrogram mean coefficient_9.28Hz',  '0_Wavelet absolute mean_1.78Hz',  '0_Wavelet absolute mean_15.98Hz',  '0_Wavelet absolute mean_2.0Hz',  '0_Wavelet absolute mean_2.28Hz',  '0_Wavelet absolute mean_2.66Hz',  '0_Wavelet absolute mean_3.2Hz',  '0_Wavelet absolute mean_4.0Hz',  '0_Wavelet absolute mean_5.33Hz',  '0_Wavelet absolute mean_7.99Hz',  '0_Wavelet energy_1.78Hz',  '0_Wavelet energy_15.98Hz',  '0_Wavelet energy_2.0Hz',  '0_Wavelet energy_2.28Hz',  '0_Wavelet energy_2.66Hz',  '0_Wavelet energy_3.2Hz',  '0_Wavelet energy_4.0Hz',  '0_Wavelet energy_5.33Hz',  '0_Wavelet energy_7.99Hz',  '0_Wavelet entropy',  '0_Wavelet standard deviation_1.78Hz',  '0_Wavelet standard deviation_15.98Hz',  '0_Wavelet standard deviation_2.0Hz',  '0_Wavelet standard deviation_2.28Hz',  '0_Wavelet standard deviation_2.66Hz',  '0_Wavelet standard deviation_3.2Hz',  '0_Wavelet standard deviation_4.0Hz',  '0_Wavelet standard deviation_5.33Hz',  '0_Wavelet standard deviation_7.99Hz',  '0_Wavelet variance_1.78Hz',  '0_Wavelet variance_15.98Hz',  '0_Wavelet variance_2.0Hz',  '0_Wavelet variance_2.28Hz',  '0_Wavelet variance_2.66Hz',  '0_Wavelet variance_3.2Hz',  '0_Wavelet variance_4.0Hz',  '0_Wavelet variance_5.33Hz',  '0_Wavelet variance_7.99Hz',  'value_count',  'value_median',  'value_mean',  'value_qmean',  'value_max',  'value_min',  'value_maxmin',  'value_std',  'value_var',  'value_diffstd',  'value_diffvar',  'value_diffmax',  'value_diffmin',  'value_diffmean',  'value_diffqmean',  'value_diffmedian',  'value_diffmaxmin',  'time_diffmean',  'time_diffqmean',  'time_diffmax',  'time_diffmin',  'time_diffmedian',  'time_diffstd',  'time_diffvar',  'time_burstiness',  'time_total',  'time_event_density',  'time_entropy',  'time_slope']
 
newrow1column = ['0_Absolute energy',  '0_Area under the curve',  '0_Autocorrelation',  '0_Average power',  '0_Centroid',  '0_ECDF Percentile Count_0',  '0_ECDF Percentile Count_1',  '0_ECDF Percentile_0',  '0_ECDF Percentile_1',  '0_ECDF_0',  '0_ECDF_1',  '0_ECDF_2',  '0_ECDF_3',  '0_ECDF_4',  '0_ECDF_5',  '0_ECDF_6',  '0_Entropy',  '0_Histogram mode',  '0_Interquartile range',  '0_Kurtosis',  '0_Max',  '0_Mean',  '0_Mean absolute deviation',  '0_Mean absolute diff',  '0_Mean diff',  '0_Median',  '0_Median absolute deviation',  '0_Median absolute diff',  '0_Median diff',  '0_Min',  '0_Negative turning points',  '0_Neighbourhood peaks',  '0_Peak to peak distance',  '0_Positive turning points',  '0_Root mean square',  '0_Signal distance',  '0_Skewness',  '0_Slope',  '0_Standard deviation',  '0_Sum absolute diff',  '0_Variance',  '0_Zero crossing rate']

newrow2column = ['0_Fundamental frequency',  '0_Human range energy',  '0_Max power spectrum',  '0_Maximum frequency',  '0_Median frequency',  '0_Power bandwidth',  '0_Spectral centroid',  '0_Spectral decrease',  '0_Spectral distance',  '0_Spectral entropy',  '0_Spectral kurtosis',  '0_Spectral positive turning points',  '0_Spectral roll-off',  '0_Spectral roll-on',  '0_Spectral skewness',  '0_Spectral slope',  '0_Spectral spread',  '0_Spectral variation',  '0_Spectrogram mean coefficient_0.0Hz',  '0_Spectrogram mean coefficient_1.03Hz',  '0_Spectrogram mean coefficient_10.31Hz',  '0_Spectrogram mean coefficient_11.34Hz',  '0_Spectrogram mean coefficient_12.37Hz',  '0_Spectrogram mean coefficient_13.4Hz',  '0_Spectrogram mean coefficient_14.44Hz',  '0_Spectrogram mean coefficient_15.47Hz',  '0_Spectrogram mean coefficient_16.5Hz',  '0_Spectrogram mean coefficient_17.53Hz',  '0_Spectrogram mean coefficient_18.56Hz',  '0_Spectrogram mean coefficient_19.59Hz',  '0_Spectrogram mean coefficient_2.06Hz',  '0_Spectrogram mean coefficient_20.62Hz',  '0_Spectrogram mean coefficient_21.65Hz',  '0_Spectrogram mean coefficient_22.68Hz',  '0_Spectrogram mean coefficient_23.72Hz',  '0_Spectrogram mean coefficient_24.75Hz',  '0_Spectrogram mean coefficient_25.78Hz',  '0_Spectrogram mean coefficient_26.81Hz',  '0_Spectrogram mean coefficient_27.84Hz',  '0_Spectrogram mean coefficient_28.87Hz',  '0_Spectrogram mean coefficient_29.9Hz',  '0_Spectrogram mean coefficient_3.09Hz',  '0_Spectrogram mean coefficient_30.93Hz',  '0_Spectrogram mean coefficient_31.97Hz',  '0_Spectrogram mean coefficient_4.12Hz',  '0_Spectrogram mean coefficient_5.16Hz',  '0_Spectrogram mean coefficient_6.19Hz',  '0_Spectrogram mean coefficient_7.22Hz',  '0_Spectrogram mean coefficient_8.25Hz',  '0_Spectrogram mean coefficient_9.28Hz',  '0_Wavelet absolute mean_1.78Hz',  '0_Wavelet absolute mean_15.98Hz',  '0_Wavelet absolute mean_2.0Hz',  '0_Wavelet absolute mean_2.28Hz',  '0_Wavelet absolute mean_2.66Hz',  '0_Wavelet absolute mean_3.2Hz',  '0_Wavelet absolute mean_4.0Hz',  '0_Wavelet absolute mean_5.33Hz',  '0_Wavelet absolute mean_7.99Hz',  '0_Wavelet energy_1.78Hz',  '0_Wavelet energy_15.98Hz',  '0_Wavelet energy_2.0Hz',  '0_Wavelet energy_2.28Hz',  '0_Wavelet energy_2.66Hz',  '0_Wavelet energy_3.2Hz',  '0_Wavelet energy_4.0Hz',  '0_Wavelet energy_5.33Hz',  '0_Wavelet energy_7.99Hz',  '0_Wavelet entropy',  '0_Wavelet standard deviation_1.78Hz',  '0_Wavelet standard deviation_15.98Hz',  '0_Wavelet standard deviation_2.0Hz',  '0_Wavelet standard deviation_2.28Hz',  '0_Wavelet standard deviation_2.66Hz',  '0_Wavelet standard deviation_3.2Hz',  '0_Wavelet standard deviation_4.0Hz',  '0_Wavelet standard deviation_5.33Hz',  '0_Wavelet standard deviation_7.99Hz',  '0_Wavelet variance_1.78Hz',  '0_Wavelet variance_15.98Hz',  '0_Wavelet variance_2.0Hz',  '0_Wavelet variance_2.28Hz',  '0_Wavelet variance_2.66Hz',  '0_Wavelet variance_3.2Hz',  '0_Wavelet variance_4.0Hz',  '0_Wavelet variance_5.33Hz',  '0_Wavelet variance_7.99Hz']

# init empty csv, we will using margin with length of 1/n
log(f"Initialize empty output csv files.")
for i in args.n_splits:
    for j in range(1, i+1):
        pd.DataFrame(columns=columnlist_test).to_csv(f"{folder_name}/train_features_split{j}_{i}_v3.csv", index=False)


def get_features(data_df):
    #data_df = data_df.copy()
    data_df['timestamp'] = data_df['timestamp']-data_df.timestamp.iloc[0]
    data_df['valdiff'] = data_df['value'].diff(1)
    data_df['timediff'] = data_df['timestamp'].diff(1)
    value_count = data_df['value'].count()
    value_median = data_df['value'].median()
    value_mean = data_df['value'].mean()
    value_qmean = data_df['value'].quantile([.25, .75]).mean()
    value_max = data_df['value'].max()
    value_min = data_df['value'].min()    
    value_maxmin = value_max - value_min
    value_std = data_df['value'].std() 
    value_var = data_df['value'].var() 
    
    value_diffmax = data_df['valdiff'].dropna().max()
    value_diffmin = data_df['valdiff'].dropna().min()
    value_diffmean  = data_df['valdiff'].dropna().mean()
    value_diffqmean = data_df['valdiff'].dropna().quantile([.25, .75]).mean()
    value_diffmedian = data_df['valdiff'].dropna().median()
    value_diffmaxmin = value_diffmax - value_diffmin
    value_diffstd = data_df['valdiff'].dropna().std() 
    value_diffvar = data_df['valdiff'].dropna().var() 
    
    time_diffmean = data_df['timediff'].dropna().mean()
    time_diffqmean = data_df['timediff'].dropna().quantile([.25, .75]).mean()
    time_diffmax = data_df['timediff'].dropna().max()
    time_diffmin = data_df['timediff'].dropna().min()
    time_diffmedian = data_df['timediff'].dropna().median()

    time_diffstd = data_df['timediff'].dropna().std()
    time_diffvar = data_df['timediff'].dropna().var()
    if len(data_df['timediff']) > 1:
        mean_diff = time_diffmean
        std_diff = time_diffstd
        time_burstiness = (std_diff - mean_diff) / (std_diff + mean_diff) if (std_diff + mean_diff) != 0 else 0
    else:
        time_burstiness = 0
        
    time_total = data_df['timestamp'].iloc[-1] - data_df['timestamp'].iloc[0]
    time_event_density = len(data_df['timestamp']) / time_total if time_total > 0 else 0
    
    if len(data_df['timediff']) > 1:
        time_diffs_prob = data_df['timediff'] / np.sum(data_df['timediff'])  # Normalize
        time_entropy = -np.sum(time_diffs_prob * np.log2(time_diffs_prob + 1e-9))
    else:
        time_entropy = 0
    time_slope = linregress(data_df['timestamp'].index, data_df['timestamp']).slope

    extracted_features = {
        "value_count": value_count,
        "value_median": value_median,
        "value_mean": value_mean,
        "value_qmean": value_qmean,
        "value_max": value_max,
        "value_min": value_min,
        "value_maxmin": value_maxmin,
        "value_std": value_std,
        "value_var": value_var,
        "value_diffstd": value_diffstd,
        "value_diffvar": value_diffvar,
        "value_diffmax": value_diffmax,
        "value_diffmin": value_diffmin,
        "value_diffmean": value_diffmean,
        "value_diffqmean": value_diffqmean,
        "value_diffmedian": value_diffmedian,
        "value_diffmaxmin": value_diffmaxmin,
        "time_diffmean": time_diffmean,
        "time_diffqmean": time_diffqmean,
        "time_diffmax": time_diffmax,
        "time_diffmin": time_diffmin,
        "time_diffmedian": time_diffmedian,

        'time_diffstd': time_diffstd,
        'time_diffvar': time_diffvar,
        'time_burstiness': time_burstiness,
        'time_total': time_total,
        'time_event_density': time_event_density,
        'time_entropy': time_entropy,
        'time_slope': time_slope
    }
    
    timestamps1 = np.linspace(data_df['timestamp'].min(), data_df['timestamp'].max(), num=len(data_df))
    timestamps2 = np.arange(data_df['timestamp'].min(), data_df['timestamp'].max(), dt)
    # Interpolate values
    interpolator = interp1d(data_df['timestamp'], data_df['value'], kind='nearest')
    values_fixed = interpolator(timestamps1)
    values_forfreq = interpolator(timestamps2)
    # Create a new DataFrame for regularized data_df
    datafix1 = pd.DataFrame({'timestamp': timestamps1, 'value': values_fixed})
    datafix2 = pd.DataFrame({'timestamp': timestamps2, 'value': values_forfreq})
    #print(regular_df.head())

    try:
        new_row1 = tsfel.time_series_features_extractor(cfg_file1, 
                                                   datafix1['value'].values, 
                                                   fs=1/((timestamps1[1]-timestamps1[0])/3600), verbose=False
                                                  )
    except:
        new_row1 = pd.DataFrame([[0.0] * len(newrow1column)], columns=newrow1column)
    try:
        new_row2 = tsfel.time_series_features_extractor(cfg_file2, 
                                                   datafix2['value'].values, 
                                                   fs=1/((timestamps2[1]-timestamps2[0])/3600), verbose=False
                                                  )
    except:
        new_row2 = pd.DataFrame([[0.0] * len(newrow2column)], columns=newrow2column)
        
    new_row3 = pd.DataFrame([extracted_features])
    new_row = pd.concat([new_row1, new_row2, new_row3], axis=1)
    columns_to_change = ['0_Kurtosis', '0_Skewness']
    new_row[columns_to_change] = new_row[columns_to_change].fillna(0)
    
    return new_row


import concurrent.futures
log(f"Computing features..")
def process_row(i):
    l_pkl = pickle.loads(zipf.read(f'train_X/{train_y.filename[i]}'))
    data = pd.DataFrame({
        'timestamp': l_pkl['t'],  # Convert 't' to timedelta
        'value': l_pkl['v']
    })
    data['timestamp'] = data['timestamp'].dt.total_seconds()
    
    for j in args.n_splits:
        for k in range(1, j+1):
            if len(data)//j>20 and j>1:
                start_idx = (k-1)*(len(data)//j) #np.random.randint(0, len(data)-len(data)//j)
                data_split = data.iloc[start_idx:start_idx+len(data)//j].copy()
                new_row = get_features(data_split)
            else:
                start_idx = 0
                new_row = get_features(data)
        
            new_row.index = [i]
            new_row.to_csv(f"{folder_name}/train_features_split{k}_{j}_v3.csv", mode='a', header=False)
        
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_row, range(0, len(train_y))), total=len(train_y)))

column_list_full = ['0_Absolute energy', '0_Area under the curve', '0_Autocorrelation', '0_Average power', '0_Centroid', '0_ECDF Percentile Count_0', '0_ECDF Percentile Count_1', '0_ECDF Percentile_0', '0_ECDF Percentile_1', '0_ECDF_0', '0_ECDF_1', '0_ECDF_2', '0_ECDF_3', '0_ECDF_4', '0_ECDF_5', '0_ECDF_6', '0_ECDF_7', '0_ECDF_8', '0_ECDF_9', '0_Entropy', '0_Histogram mode', '0_Interquartile range', '0_Kurtosis', '0_Max', '0_Mean', '0_Mean absolute deviation', '0_Mean absolute diff', '0_Mean diff', '0_Median', '0_Median absolute deviation', '0_Median absolute diff', '0_Median diff', '0_Min', '0_Negative turning points', '0_Neighbourhood peaks', '0_Peak to peak distance', '0_Positive turning points', '0_Root mean square', '0_Signal distance', '0_Skewness', '0_Slope', '0_Standard deviation', '0_Sum absolute diff', '0_Variance', '0_Zero crossing rate', '0_Fundamental frequency', '0_Human range energy', '0_Max power spectrum', '0_Maximum frequency', '0_Median frequency', '0_Power bandwidth', '0_Spectral centroid', '0_Spectral decrease', '0_Spectral distance', '0_Spectral entropy', '0_Spectral kurtosis', '0_Spectral positive turning points', '0_Spectral roll-off', '0_Spectral roll-on', '0_Spectral skewness', '0_Spectral slope', '0_Spectral spread', '0_Spectral variation', '0_Spectrogram mean coefficient_0.0Hz', '0_Spectrogram mean coefficient_1.03Hz', '0_Spectrogram mean coefficient_10.31Hz', '0_Spectrogram mean coefficient_11.34Hz', '0_Spectrogram mean coefficient_12.37Hz', '0_Spectrogram mean coefficient_13.4Hz', '0_Spectrogram mean coefficient_14.44Hz', '0_Spectrogram mean coefficient_15.47Hz', '0_Spectrogram mean coefficient_16.5Hz', '0_Spectrogram mean coefficient_17.53Hz', '0_Spectrogram mean coefficient_18.56Hz', '0_Spectrogram mean coefficient_19.59Hz', '0_Spectrogram mean coefficient_2.06Hz', '0_Spectrogram mean coefficient_20.62Hz', '0_Spectrogram mean coefficient_21.65Hz', '0_Spectrogram mean coefficient_22.68Hz', '0_Spectrogram mean coefficient_23.72Hz', '0_Spectrogram mean coefficient_24.75Hz', '0_Spectrogram mean coefficient_25.78Hz', '0_Spectrogram mean coefficient_26.81Hz', '0_Spectrogram mean coefficient_27.84Hz', '0_Spectrogram mean coefficient_28.87Hz', '0_Spectrogram mean coefficient_29.9Hz', '0_Spectrogram mean coefficient_3.09Hz', '0_Spectrogram mean coefficient_30.93Hz', '0_Spectrogram mean coefficient_31.97Hz', '0_Spectrogram mean coefficient_4.12Hz', '0_Spectrogram mean coefficient_5.16Hz', '0_Spectrogram mean coefficient_6.19Hz', '0_Spectrogram mean coefficient_7.22Hz', '0_Spectrogram mean coefficient_8.25Hz', '0_Spectrogram mean coefficient_9.28Hz', '0_Wavelet absolute mean_1.78Hz', '0_Wavelet absolute mean_15.98Hz', '0_Wavelet absolute mean_2.0Hz', '0_Wavelet absolute mean_2.28Hz', '0_Wavelet absolute mean_2.66Hz', '0_Wavelet absolute mean_3.2Hz', '0_Wavelet absolute mean_4.0Hz', '0_Wavelet absolute mean_5.33Hz', '0_Wavelet absolute mean_7.99Hz', '0_Wavelet energy_1.78Hz', '0_Wavelet energy_15.98Hz', '0_Wavelet energy_2.0Hz', '0_Wavelet energy_2.28Hz', '0_Wavelet energy_2.66Hz', '0_Wavelet energy_3.2Hz', '0_Wavelet energy_4.0Hz', '0_Wavelet energy_5.33Hz', '0_Wavelet energy_7.99Hz', '0_Wavelet entropy', '0_Wavelet standard deviation_1.78Hz', '0_Wavelet standard deviation_15.98Hz', '0_Wavelet standard deviation_2.0Hz', '0_Wavelet standard deviation_2.28Hz', '0_Wavelet standard deviation_2.66Hz', '0_Wavelet standard deviation_3.2Hz', '0_Wavelet standard deviation_4.0Hz', '0_Wavelet standard deviation_5.33Hz', '0_Wavelet standard deviation_7.99Hz', '0_Wavelet variance_1.78Hz', '0_Wavelet variance_15.98Hz', '0_Wavelet variance_2.0Hz', '0_Wavelet variance_2.28Hz', '0_Wavelet variance_2.66Hz', '0_Wavelet variance_3.2Hz', '0_Wavelet variance_4.0Hz', '0_Wavelet variance_5.33Hz', '0_Wavelet variance_7.99Hz', 'value_count', 'value_median', 'value_mean', 'value_qmean', 'value_max', 'value_min', 'value_maxmin', 'value_std', 'value_var', 'value_diffstd', 'value_diffvar', 'value_diffmax', 'value_diffmin', 'value_diffmean', 'value_diffqmean', 'value_diffmedian', 'value_diffmaxmin', 'time_diffmean', 'time_diffqmean', 'time_diffmax', 'time_diffmin', 'time_diffmedian', 'time_diffstd', 'time_diffvar', 'time_burstiness', 'time_total', 'time_event_density', 'time_entropy', 'time_slope']
log(f"Fixing files..")
#list generated files
listcsv = []
for files_list in os.listdir(folder_name):
    if ('train' in files_list) and ('features_margin_' in files_list) and (os.path.splitext(files_list)[-1] == '.csv'):
        listcsv.append('./'+folder_name+'/'+files_list)

# fixer
print('')
for filename in tqdm(listcsv):
    traindat = pd.read_csv(filename, index_col=0)
    columns_with_nan = traindat.columns[traindat.isnull().any()].tolist()
    traindat = traindat.dropna()
    traindat = traindat[~traindat.index.duplicated(keep='first')].sort_index()
    traindat['index'] = traindat.index
    traindat['indexdiff'] = traindat['index'].diff(1)
    list_i_error = [int(x)-1 for x in traindat[traindat['indexdiff'] != 1.0].index.tolist()[1:]]
    for i in list_i_error:
        #l_pkl = pickle.loads(zipf.read(f'train_X/{train_y.filename[i]}'))
        l_pkl = pickle.loads(zipftest.read(listtestfile[i]))
        data = pd.DataFrame({
            'timestamp': l_pkl['t'],  # Convert 't' to timedelta
            'value': l_pkl['v']
        })
        data['timestamp'] = data['timestamp'].dt.total_seconds()
        new_row = get_features(data)
        new_row = new_row.fillna(0.0)
        new_row.index = [i]
        for col in column_list_full:
            if col not in new_row.columns:
                new_row[col] = 0.0
        new_row = new_row[column_list_full]
        new_row.to_csv(filename, mode='a', header=False)
    
    traindat = pd.read_csv(filename, index_col=0)
    traindat_cleaned = traindat.dropna()
    traindat_cleaned = traindat_cleaned[~traindat_cleaned.index.duplicated(keep='first')].sort_index()
    
    #check if len is the same as original n of data
    if len(traindat_cleaned)==31839:
        traindat_cleaned.to_csv(filename.replace(folder_name, folder_name2))
    else:
        print('len weird', filename)
        
log(f"Training features extracted. File location: {folder_name2}")
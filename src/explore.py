#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data

# --- Configuration starts here --------------------------------

# What diagrams to plot
CONFIG = {
    "outliers"      : False,
    "scatter_matrix": False,
    "correlation"   : False,
    "simple"        : True,
    "air_quality"   : False,
    "temperature"   : False,
    "compare"       : False,
}

# 'EXT', 'IN1' or 'IN2'
sensor = "EXT"

# Select which variables to plot (for 'simple' plot only)
variables = { 
    "BME680"     : False,   # COV (Bosch sensor)
    "co2"        : False,    # CO2
    "humidity"   : False,    # Humidity
    "noxi"       : False,   # NOx (Nitrogen oxide) index
    "noxr"       : False,   # NOx raw data
    "pm10p0"     : False,   # 10.0 µm particles 
    "pm1p0"      : False,   # 1.0 µm particles
    "pm2p5"      : False,   # 2.5 µm particles
    "pm4p0"      : False,   # 4.0 µm particles
    "temperature": False,    # Temperature (C°)
    "voxi"       : True,   # COV index (Sensirion sensor)
    "voxr"       : False,   # COV raw
    "sfa30"      : False,    # COV (Sensirion sensor, different technology than BME & vox)
}

# Start and end timestamps for the data to be plotted
# NOTE: Needs debugging
# RANGE_START = None
# RANGE_END = None
# RANGE_START = pd.Timestamp(2024, 2, 6, 12, 0, 0)
RANGE_START = pd.Timestamp(2024, 3, 15, 12, 0, 0)
RANGE_END = pd.Timestamp(2024, 2, 9, 12, 0, 0)

# Remove outliers
DO_REMOVE_OUTLIERS = True
DO_REMOVE_GRADIENT_OUTLIERS = True

# --- Configuration ends here ----------------------------------

plots = { 
    "BME680"     : None, 
    "co2"        : None, 
    "humidity"   : None, 
    "noxi"       : None,
    "noxr"       : None,
    "pm10p0"     : None,
    "pm1p0"      : None,
    "pm2p5"      : None,
    "pm4p0"      : None,
    "temperature": None,
    "voxi"       : None,
    "voxr"       : None,
    "sfa30"      : None,
}

COLORS = {0: '#39B185', 1: '#9CCB86', 2: '#E9E29C', 3: '#EEB479', 4: '#E88471', 5: '#CF597E', 6: '#222222', 7: 'white'}
COLOR_COUNT = len(COLORS) - 2

plot_count = sum(variables.values())    # Only counts True
plot_index = 1
root_plot = None
POINT_COUNT = 0      # 0 for all

def create_plot(data, y, color=None):
    global root_plot
    global plots
    global plot_index
    global plot_count
    global POINT_COUNT

    plot = None
    if root_plot is None:
        plot = plt.subplot(plot_count, 1, plot_index)
        root_plot = plot
    else:
        plot = plt.subplot(plot_count, 1, plot_index, sharex=root_plot)

    plt.title(y)
    plt.plot(data.index[-POINT_COUNT:], data[y][-POINT_COUNT:], color=color)

    plot_index += 1
    plots[y] = plot

    return plot


# Load data and select the sensor
dataframes = data.load_data(RANGE_START, RANGE_END)
df = dataframes[sensor]
df_length = len(df.index)

# Print information
print(f"Sensor: {sensor}")
print(f"Number of points in data: {df_length}")
data_total_duration = df.index[-1] - df.index[0]
print(f"Time interval: {df.index[0]} - {df.index[-1]} ({data_total_duration})")
missing_data_total_duration = df["co2"].isna().sum() * pd.Timedelta(minutes=1)     # If co2 is missing, then other variables may as well
print(f"Total time of missing data: {missing_data_total_duration} ({missing_data_total_duration / data_total_duration * 100} %)")

# Remove outliers
if DO_REMOVE_OUTLIERS:
    outliers = data.remove_outliers(df)
    outliers_count = outliers.notna().sum().sum()
    outliers_total_duration = outliers_count / len(outliers.columns) * pd.Timedelta(minutes=1)
    print(f"Number of outliers: {outliers_count}")
    print(f"Total time of outliers: {outliers_total_duration} ({outliers_total_duration / data_total_duration * 100} %)")
    # Plot outliers
    if CONFIG["outliers"] and outliers_count > 0:
        print("Creating outliers diagram...")
        fig, ax = plt.subplots(1, sharex=True)
        for i in range(len(outliers.columns)):
            ax.plot(df.index, outliers.iloc[:, i] * (i + 1), label=outliers.columns[i])
        ax.legend()
        plt.show()

if DO_REMOVE_GRADIENT_OUTLIERS:
    # Remove temperature outliers
    temperature_outliers = data.remove_gradient_outliers(df, 'temperature')
    temperature_outliers_count = temperature_outliers.notna().sum().sum()
    temperature_outliers_total_duration = temperature_outliers_count * pd.Timedelta(minutes=1)
    print(f"Number of temperature outliers: {temperature_outliers_count}")
    print(f"Total time of temperature outliers: {temperature_outliers_total_duration} ({temperature_outliers_total_duration / data_total_duration * 100} %)")
    # Remove humidity outliers
    humidity_outliers = data.remove_gradient_outliers(df, 'humidity')
    humidity_outliers_count = humidity_outliers.notna().sum().sum()
    humidity_outliers_total_duration = humidity_outliers_count * pd.Timedelta(minutes=1)
    print(f"Number of humidity outliers: {humidity_outliers_count}")
    print(f"Total time of humidity outliers: {humidity_outliers_total_duration} ({humidity_outliers_total_duration / data_total_duration * 100} %)")


# Scatter diagram diagram
def plot_scatter_matrix():
    print("Creating scatter matrix...")

    # Use only the variables that are set to True
    axes = pd.plotting.scatter_matrix(
        df[[key for key, value in variables.items() if value]], 
        alpha=0.1)
    
    print("Showing...")
    #plt.tight_layout()
    plt.savefig('figures/scatter_diagram.png', dpi=600)
    plt.show()
    plt.close()

# Correlation information
def calculate_correlation():
    print("Calculating correlation...")

    # Calculate Pearson's correlation coefficient
    # Pearson’s correlation coefficient is calculated by dividing the covariance of two variables 
    # by the product of their respective standard deviations.
    # Basically, it is the normalization of the covariance between two variables to give an interpretable score.
    correlation = df.corr(method='pearson')
    print("Pearson correlation:")
    print(correlation)

    # Calculate Spearman's correlation coefficient
    # "monotone"?"
    correlation = df.corr(method='spearman')
    print("Spearman correlation:")
    print(correlation)


# Simple diagram
def plot_simple_diagram():
    print("Creating simple diagram...")

    # Color
    color_index = 0
    colors = COLORS
    if plot_count == 2:
        colors = {0: COLORS[0], 1: COLORS[COLOR_COUNT - 1]}
    elif plot_count == 3:
        colors = {0: COLORS[0], 1: COLORS[2 % COLOR_COUNT], 2: COLORS[4 % COLOR_COUNT]}

    for variable in variables.keys():
        if variables[variable]:
            create_plot(df, variable, colors[color_index])
            color_index = color_index + 1 % COLOR_COUNT 

    print("Showing...")
    plt.show()

# Air quality diagram
def plot_air_quality_diagram():

    print("Creating air quality diagram...")
    fig, ax = plt.subplots(1, sharex=True) 

    class Timeline:
        def __init__(self, name: str, y: float, height: float, children: list):
            self.name = name
            self.y = y
            self.height = height
            self.children = children
            self.bars = []
            self.categories = []
            self.colors = []

    # y value and height for each variable
    master_timeline = Timeline("global", -0.2, .8, [
        Timeline("co2", -1, .4, None), 
        Timeline("pm2p5", -1.5, .4, None), 
        Timeline("COV", -2, .4, [
            Timeline("BME680", -7/3, .3, None), 
            Timeline("sfa30", -8/3, .3, None), 
            Timeline("voxi", -3, .3, None), 
        ]), 
    ])

    # Create new 'category' columns
    df['co2_cat'] = pd.cut(df['co2'], bins=[0, 600, 1000, 1200, 1700, 2500, np.inf], labels=[0, 1, 2, 3, 4, 5])
    df['pm2p5_cat'] = pd.cut(df['pm2p5'], bins=[0, 25, 50, 75, 100, 250, np.inf], labels=[0, 1, 2, 3, 4, 5])
    df['BME680_cat'] = pd.cut(df['BME680'], bins=[0, 200000, 400000, 600000, 800000, 1000000, np.inf], labels=[0, 1, 2, 3, 4, 5])
    df['sfa30_cat'] = pd.cut(df['sfa30'], bins=[0, 10, 20, 30, 40, 50, np.inf], labels=[0, 1, 2, 3, 4, 5])
    df['voxi_cat'] = pd.cut(df['voxi'], bins=[0, 1000, 2000, 3000, 4000, 5000, np.inf], labels=[0, 1, 2, 3, 4, 5])

    # Reccursively calculate each timeline in reverse order (children first)
    def display_timeline(timeline: Timeline) -> np.ndarray:
        categories = None

        # If timeline has children, then its category is the max of the child categories
        if timeline.children is not None:
            categories = np.zeros(df_length, dtype=float)   # Using float to allow NaN
            for child in timeline.children:
                child_categories = display_timeline(child)
                categories = np.maximum(categories, child_categories)
        else:
            categories = df[timeline.name + '_cat'].astype(float).values
            # Handle NaN values
            categories[np.isnan(categories) & outliers[timeline.name].notna()] = 6  # Outliers
            categories[np.isnan(categories)] = 7    # Undefined
        
        # Print categories and there count
        print(f"{timeline.name} categories: {np.unique(categories, return_counts=True)}")

        # Get category slices
        diff = np.concatenate(([0], np.diff(categories)))
        slices = np.ma.clump_unmasked(np.ma.masked_where(diff != 0, categories))

        # Get diagram data
        bar_array = []
        category_array = []
        color_array = []
        for s in slices:
            start = df.index[s.start]
            duration = df.index[(s.stop if s.stop < df_length else -1)] - start
            category = categories[s.start]
            bar_array.append((start, duration))
            category_array.append(category)
            # Choose color based on category
            color_array.append(COLORS[category])
        

        # Diagram is a broken bar plot
        ax.broken_barh(bar_array, (timeline.y, timeline.height), facecolors=color_array)

        # Add bar name
        ax.text(df.index[0], timeline.y + timeline.height / 2, timeline.name, ha='right', va='center')

        return categories 
                

    # Recursively display master timeline and its children
    display_timeline(master_timeline)

    # Add legend
    legend = []
    for key, value in COLORS.items():
        legend.append(plt.Rectangle((0, 0), 1, 1, fc=value))
    # Hide some axis
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Legend
    legend_keys = list(COLORS.keys())
    legend_keys[6] = 'outliers'
    legend_keys[7] = 'undefined'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(legend, legend_keys, loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot diagram
    plt.show()


# Temperature diagram
# Displays temperature with cleaned temperature, then its gradient
# NOTE: This is the outdated version of temperature filtering
def plot_temperature_diagram():
    print("Creating temperature diagram...")

    fig, axs = plt.subplots(2, sharex=True)

    # Plot temperature
    axs[0].plot(df.index, df['temperature'], label='Temperature')

    # Temperature gradient
    temperature_gradient = df['temperature'].diff()

    # Disable false positive warning
    pd.options.mode.chained_assignment = None  # default='warn'
    
    # Cleaned temperature removes noise
    # Noise is detected when gradient is bigger than 1, we then remove the 50 next minutes
    correction_end = pd.Timestamp(0)
    df['temperature_cleaned'] = df['temperature'].copy()
    for i in range(len(df['temperature']) - 1):
        # If the gradient is too big, we remove the next 50 minutes
        if abs(temperature_gradient.iloc[i]) > 1:
            correction_end = df.index[i] + pd.Timedelta(minutes=50)
        # Remove point if it is in the correction period
        if df.index[i] < correction_end:
            df['temperature_cleaned'].iloc[i] = np.nan

    # Enable back the warning
    pd.options.mode.chained_assignment = 'warn'

    # Interplolate the cleaned temperature
    df['temperature_cleaned'] = df['temperature_cleaned'].interpolate()

    # Plot cleaned temperature
    axs[0].plot(df.index, df['temperature_cleaned'], label='Cleaned temperature')

    # Plot smooth cleaned temperature
    df['temperature_cleaned_smooth'] = df['temperature_cleaned'].rolling(window=10).mean()
    axs[0].plot(df.index, df['temperature_cleaned_smooth'], label='Smoothed cleaned temperature')

    # New plot for cleaned temperature gradient (calculation is based on time)
    axs[1].plot(
        df.index, 
        df["temperature_cleaned_smooth"].diff() / df.index.to_series().diff().dt.total_seconds(), 
        label='Temperature gradient'
    )

    axs[0].legend()
    axs[1].legend()
    plt.show()

# Compare IN1 & IN2
def plot_comparison_diagram():
    df_in1 = dataframes["IN1"]
    df_in2 = dataframes["IN2"]
    # Correlation between IN1 & IN2 (only variable voxi)
    print("Calculating correlation between the variables 'voxi' of IN1 & IN2...")
    print("Pearson correlation:")
    print(df_in1['voxi'].corr(df_in2['voxi'], method='pearson'))
    print("Spearman correlation:")
    print(df_in1['voxi'].corr(df_in2['voxi'], method='spearman'))

    print("Creating comparison diagram...")
    fig, axs = plt.subplots(2, sharex=True, sharey=True)

    # Plot VOX
    axs[0].plot(df_in1.index, df_in1['voxi'], label='VOX 0')
    axs[1].plot(df_in2.index, df_in2['voxi'], label='VOX 1')

    plt.show()


# Main
if __name__ == "__main__":
    if CONFIG["scatter_matrix"]:
        plot_scatter_matrix()
    if CONFIG["correlation"]:
        calculate_correlation()
    if CONFIG["simple"]:
        plot_simple_diagram()
    if CONFIG["air_quality"]:
        plot_air_quality_diagram()
    if CONFIG["temperature"]:
        plot_temperature_diagram()
    if CONFIG["compare"]:
        plot_comparison_diagram()

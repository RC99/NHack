import pandas as pd
import requests
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Switch to a non-GUI backend
import matplotlib.pyplot as plt
import os

def fetch_data(url):
    """Fetch data from the given URL and return as a DataFrame."""
    print("Fetching data from URL...")
    response = requests.get(url)
    if response.status_code == 200:
        print("Data fetched successfully.")
        data = response.json()
        df = pd.DataFrame(data)
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(1))  # Display the first row to verify data
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def auto_map_categorical(df):
    """Automatically map categorical columns to numeric values."""
    mappings = {}
    for col in df.columns:
        print(f"Processing column: {col}")
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            print(f"Column {col} contains dictionaries or complex types. Skipping.")
            continue
        
        print(f"Unique values in {col}:")
        print(df[col].unique())
        
        if df[col].dtype == 'object':  # Threshold for categorical data
            print(f"Identified {col} as a categorical column.")
            value_counts = df[col].value_counts()
            mappings[col] = {v: k for k, v in enumerate(value_counts.index)}
            df[col] = df[col].astype('category')  # Ensure categorical type for plotting
    return mappings

def generate_images(df, image_dir, selected_cols):
    """Generate and save histograms and bar plots for specified columns."""
    if df.empty:
        print("DataFrame is empty. No images will be generated.")
        return

    print("DataFrame columns and types:")
    print(df.dtypes)

    # Filter columns
    columns_to_plot = [col for col in selected_cols if col in df.columns]
    
    if len(columns_to_plot) == 0:
        print("No valid columns found in the DataFrame.")
        return

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Plot bar plots for selected columns
    for i, col in enumerate(columns_to_plot):
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            print(f"Generating bar plot for column: {col}")
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x=df[col])
            plt.title(f'Bar Plot of {col}')
            plt.xlabel('Category')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
            
            # Annotate bars with counts
            for p in ax.patches:
                height = p.get_height()
                if isinstance(height, float) and not pd.isna(height):
                    height = int(round(height))
                ax.annotate(format(height, 'd'), 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points')
            
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{image_dir}/barplot_{i}.png')
            plt.close()
            print(f"Saved bar plot for column {col} as barplot_{i}.png")

def generate_top_road_name_plot(df, image_dir):
    """Generate and save a horizontal bar plot for the top 5 road names with the highest counts."""
    if df.empty or 'road_name' not in df.columns:
        print("DataFrame is empty or 'road_name' column is missing. No plot will be generated.")
        return

    # Filter out NaN values and get top 5 road names with highest counts
    top_road_names = df['road_name'].dropna().value_counts().nlargest(5)
    if top_road_names.empty:
        print("No valid top road names found.")
        return

    print(f"Top 5 road names with the highest counts:")
    print(top_road_names)

    # Create a DataFrame for plotting
    top_road_names_df = top_road_names.reset_index()
    top_road_names_df.columns = ['road_name', 'count']
    
    # Ensure counts are integers
    top_road_names_df['count'] = top_road_names_df['count'].astype(int)
    print(top_road_names_df)

    # Plotting with matplotlib
    plt.figure(figsize=(12, 8))
    plt.barh(top_road_names_df['road_name'], top_road_names_df['count'], color='skyblue')
    plt.title('Top 5 Road Names with Highest Accident Counts')
    plt.xlabel('Count')
    plt.ylabel('Road Name')

    # Annotate bars with counts
    for index, value in enumerate(top_road_names_df['count']):
        plt.text(value, index, str(value), va='center', ha='left')

    plt.tight_layout()
    plt.savefig(f'{image_dir}/top_road_names.png')
    plt.close()
    print(f"Saved horizontal bar plot for top 5 road names as top_road_names.png")

def plot_crash_trends(df, image_dir):
    """Plot the number of crashes over time."""
    df['crash_date_time'] = pd.to_datetime(df['crash_date_time'], errors='coerce')
    df = df.dropna(subset=['crash_date_time'])
    df.set_index('crash_date_time', inplace=True)
    monthly_counts = df.resample('M').size()
    
    plt.figure(figsize=(12, 8))
    plt.plot(monthly_counts.index, monthly_counts, marker='o', linestyle='-')
    for x, y in zip(monthly_counts.index, monthly_counts):
        plt.text(x, y, str(y), ha='center', va='bottom')

    plt.title('Monthly Crash Trends')
    plt.xlabel('Date')
    plt.ylabel('Number of Crashes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{image_dir}/crash_trends.png')
    plt.close()
    print("Saved crash trends plot.")

def plot_crash_type_distribution(df, image_dir):
    """Plot the distribution of crash types."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='acrs_report_type', data=df)
    plt.title('Distribution of Crash Types')
    plt.xlabel('Crash Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            height = int(round(height))
            ax.annotate(format(height, 'd'), 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{image_dir}/crash_type_distribution.png')
    plt.close()
    print("Saved crash type distribution plot.")

def plot_surface_condition_distribution(df, image_dir):
    """Plot the distribution of surface conditions."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='surface_condition', data=df)
    plt.title('Distribution of Surface Conditions')
    plt.xlabel('Surface Condition')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            height = int(round(height))
            ax.annotate(format(height, 'd'), 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{image_dir}/surface_condition_distribution.png')
    plt.close()
    print("Saved surface condition distribution plot.")

def plot_weather_conditions(df, image_dir):
    """Plot the distribution of weather conditions."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='weather', data=df)
    plt.title('Distribution of Weather Conditions')
    plt.xlabel('Weather')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        if pd.notnull(height):
            height = int(round(height))
            ax.annotate(format(height, 'd'), 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{image_dir}/weather_conditions.png')
    plt.close()
    print("Saved weather condition distribution plot.")



def plot_crash_count_by_route_type(df, image_dir):
    """Plot the number of crashes by route type."""
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(y='route_type', data=df, order=df['route_type'].value_counts().index)
    plt.title('Number of Crashes by Route Type')
    plt.xlabel('Number of Crashes')
    plt.ylabel('Route Type')

    # Annotate bars with counts
    for p in ax.patches:
        width = p.get_width()
        if pd.notnull(width):
            width = int(round(width))
            ax.annotate(format(width, 'd'), 
                (width, p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', 
                xytext=(5, 0), 
                textcoords='offset points')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{image_dir}/crash_count_by_route_type.png')
    plt.close()
    print("Saved crash count by route type plot.")


def setup_images(image_dir):
    """Fetch data, map categorical columns, and generate images."""
    url = 'https://data.montgomerycountymd.gov/resource/bhju-22kf.json?$limit=5000'
    df = fetch_data(url)
    if not df.empty:
        auto_map_categorical(df)
        selected_columns = ['acrs_report_type', 'hit_run', 'road_grade', 'municipality', 'surface_condition']
        print("Selected columns for plotting:")
        print(selected_columns)
        print("Columns available in DataFrame:")
        print(df.columns.tolist())
        generate_images(df, image_dir, selected_columns)
        generate_top_road_name_plot(df, image_dir)
        plot_crash_trends(df, image_dir)
        plot_crash_type_distribution(df, image_dir)
        plot_surface_condition_distribution(df, image_dir)
        plot_weather_conditions(df, image_dir)
        plot_crash_count_by_route_type(df, image_dir)

# Define the image directory
#image_dir = '/Users/reetvikchatterjee/NvidiaHack/images'
#setup_images(image_dir)

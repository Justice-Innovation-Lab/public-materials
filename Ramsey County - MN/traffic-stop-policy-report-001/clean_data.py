"""
For our analysis, we received two datasets from Ramsey County ECC. One
contains all calls for service from January 2016 through September 2022.
The other contains all traffic stop calls for service from January 2016
through September 2022. Though the traffic stop data is a subset of the
all calls data, it contains an additional text column with the 'comment'
from the officer that details the reason for the stop, the race and
gender of the individual stopped, and whether the car and person
were searched.

The code below details the cleaning and manipulation of these datasets,
which was done in both python and R for various steps. This is because 
individual members of our analysis team are more familiar with either 
python or R and were responsible for different sections of the code. 
The cleaning and manipulation was originally done in coding notebooks 
that also contain many smaller, investigative steps. The code presented 
here is a streamlined version to highlight the necessary manipulation steps.
"""

# Load packages
import geopandas as gpd
import numpy as np
import pandas as pd

# The dataset of all calls for service
df_all_calls = pd.read_csv("path_to_data.csv")

# This is an additional dataset that was provided that has every 911 call
df_911_calls = pd.read_csv("path_to_data.csv")

df_all_calls["is_911_call"] = False
df_all_calls["is_911_call"] = df_all_calls["is_911_call"].mask(
    df_all_calls["ID"].isin(df_911_calls["Master_Incident_ID"]),
    True,
)


def add_census_geoids(df: pd.DataFrame, shapefile_path: str) -> pd.DataFrame:
    """
    Takes in dataframe with latitude and longitude and adds column that contains a 2010 census GEOID.
    This GEOID allows for mapping to census block all the way up to state.

    Parameters:
    - df (DataFrame): DataFrame with latitude and longitude information for each address.
    Assumes latitude and longitude are stored in columns named 'Latitude' and 'Longitude'
    - shapefile_path (str): Path to shapefile containing census block information.

    Returns:
    - merged_df (DataFrame): DataFrame of same length as df with address information and census GEOID.
    """

    df_ = df.copy()

    census_blocks = gpd.read_file(shapefile_path)

    df_["Latitude"] = df_["Latitude"] / 1000000
    df_["Longitude"] = -1 * df_["Longitude"] / 1000000
    df_.reset_index(inplace=True, drop=False)

    # Create new geopandas dataframe from the lat lon points, converting the float values to POINT values
    gdf = gpd.GeoDataFrame(
        df_, geometry=gpd.points_from_xy(df_.Longitude, df_.Latitude, crs="EPSG:4269")
    )
    gdf = gdf.to_crs(epsg=2811)
    census_blocks = census_blocks.to_crs(epsg=2811)

    # Spatial join the POINT values to census_block POLYGON values that the latlon are found within
    merged_file = gpd.sjoin(gdf, census_blocks, how="inner", predicate="within")
    merged_df = pd.DataFrame(merged_file)

    merged_df = pd.merge(
        df_, merged_df[["GEOID10", "index"]], on="index", how="left", validate="1:1"
    )

    assert (
        merged_df.shape[0] == df_.shape[0]
    ), "Number of rows in merged_df and df do not match"

    return merged_df


shapefile_path = "MN_block_2010.shp"
df_all_calls = add_census_geoids(df_all_calls, shapefile_path)

# Cleaning and manipulation for traffic stop data. This data was originally provided as
# an excel file with multiple tabs for each year of data.
traffic_stop_data = pd.read_excel("path_to_data.xlsx")

df_traffic_stops = pd.DataFrame()

for year in range(2016, 2023):
    temp_df = df[str(year)]
    temp_df["file_year"] = year
    df_traffic_stops = pd.concat([df_traffic_stops, temp_df])

df_traffic_stops = add_census_geoids(df_traffic_stops, shapefile_path)

# Adding census demographic data to the traffic stops data
infile = "MN_blk_grp_2015to2019.csv"
df_census_demos = pd.read_csv(infile)

df_census_demos["geoid_bg"] = (
    df_census_demos["geoid"].astype(str).str.slice(start=7).astype(float)
)
df_traffic_stops["geoid_bg"] = (
    df_traffic_stops["GEOID10"].astype(str).str.slice(0, 12).astype(float)
)
df_final_traffic_stops = pd.merge(
    df_traffic_stops,
    df_census_demos,
    on="geoid_bg",
    how="left",
    validate="m:1",
    indicator=True,
)

# @Rory insert write steps here for df_final_traffic_stops and df_all_calls as parquet files

######################################################################
# Additional cleaning and manipulation is performed in R at this point.

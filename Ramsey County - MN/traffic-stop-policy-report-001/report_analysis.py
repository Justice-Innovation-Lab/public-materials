"""
The code in this file reflects the necessary code for JIL's public report on
traffic stops in Ramsey County.

This code was used to produce the aggregated datasets used for the visuals
in the report. This code does not reflect all of JIL's: initial exploratory analysis,
robustness checks under various assumptions, visualization code, nor the code
currently under development for future research publications. These code categories
were excluded in the interest of brevity, but can be provided upon valid request.
"""

from datetime import datetime
from math import ceil

# The following packages were used to produce the report's visuals and are included for reference
import altair as alt
import numpy as np
import pandas as pd
import vega
from altair import datum
from altair_saver import save

########################################################################
# Traffic Stop Analysis
########################################################################
df_traffic_stops = pd.read_parquet("code_f/data/stats_allyears_wgeoid.parquet")

# Date after which all PDs included in ECC data
ALL_PDS_START_DATE = pd.to_datetime("2018-03-08")

# Last full month of data
ALL_PDS_END_DATE = pd.to_datetime("2022-10-08")

# RCAO policy announcement date
POLICY_START_DATE = pd.to_datetime("2021-09-08")

# Create variable indicating that either the vehicle or person was searched
df_traffic_stops["is_searched"] = np.where(
    (df_traffic_stops["person_searched_1"] == True)
    | (df_traffic_stops["vehicle_searched_1"] == True),
    True,
    False,
)

# Subtract 7 days for data viz month groupings (will result in RCAO policy implementation date
# showing as the start of Sept 2021 so monthly grouping will cleanly align with whole months)
df_traffic_stops["date"] = df_traffic_stops["time_callenteredqueue"] - pd.Timedelta(
    days=7
)
ALL_PDS_START_DATE = ALL_PDS_START_DATE - pd.Timedelta(days=7)
ALL_PDS_END_DATE = ALL_PDS_END_DATE - pd.Timedelta(days=7)
POLICY_START_DATE = POLICY_START_DATE - pd.Timedelta(days=7)

# Remove timezone info
df_traffic_stops["date"] = df_traffic_stops["date"].dt.tz_localize(None)

df_ = df_traffic_stops.copy()

# Remove FGPD since unknown alignment
df_ = df_.loc[df_["vehicle_jurisdiction"] != "FGPD"]

# Add variable to group PDs in accordance with alignment to RCAO policy
df_["department"] = "unknown"
df_["department"] = np.where(
    df_["vehicle_jurisdiction"] == "SPPD", "SPPD", df_["department"]
)
df_["department"] = np.where(
    (df_["vehicle_jurisdiction"] != "SPPD") & (df_["policy"] == "Aligned"),
    "Other Aligned PDs",
    df_["department"],
)
df_["department"] = np.where(
    (df_["vehicle_jurisdiction"] != "SPPD") & (df_["policy"] != "Aligned"),
    "Unchanged Policy PDs",
    df_["department"],
)

# Rename race groups
df_["race_1"] = df_["race_1"].replace(
    {"Native Am": "Native American", "Hispanic": "Hispanic or Latino"}
)

# Limit dates of data used in analyses
df_ = df_.loc[(df_["date"] >= ALL_PDS_START_DATE) & (df_["date"] <= ALL_PDS_END_DATE)]

# Remove all stops with multiple entries.
# This is done because it is unclear whether these incidents represent two police cars
# stopping the same car, multiple cars being stopped at the same time,
# or multiple persons being stopped.
# This is a conservative approach that was tested against the counterfactual.
df_ = df_.drop_duplicates(subset=["master_incident_number"], keep=False)

# Remove race coding errors.
# Race was expected to be coded based on a numerical code for each race.
# In certain instances, this was not done and it was instead coded using a letter.
# To be conservative, rather than assume the intention of the officer, these were removed.
df_ = df_.loc[df_["race_code_1"] == "Numerically Coded"]

# Aggregate data at monthly level and find percentage of stops with search
def get_monthly_stops_searches(df, aggregating_column):
    """
    Get the per month number of stops and number of searches grouped by some categorical column of
    interest.
    Parameters:
    - df (DataFrame): DataFrame of traffic stops or 911 calls where each row is a distinct incident.
    - aggregating_column (str): name of column in df that the dataframe should be grouped on

    Returns:
    - df_returned (DataFrame): DataFrame that is grouped by the year-month of the incidents and
    includes the count of distinct stops or 911 calls in that year-month.
    """
    df_returned = (
        df.groupby([aggregating_column, pd.Grouper(key="date", freq="MS")])
        .agg(
            monthly_Stops=("master_incident_number", "nunique"),
            monthly_Searches=("is_searched", "sum"),
        )
        .reset_index()
    )

    return df_returned


# Calculate average daily stops and searches by month (accounts for different # of days per month)
def monthly_stops_and_searches(df):
    """
    Calculate the average daily number of stops and number of searches after accounting for the
    number of days in each month.
    Parameters:
    - df (DataFrame): DataFrame of all stops.

    Returns:
    - df_returned (DataFrame): DataFrame grouped by year-month and providing the average number of
    daily stops and searches.
    """
    df_returned = (
        df.groupby([pd.Grouper(key="date", freq="MS")])
        .agg(
            count_Stops=("master_incident_number", "nunique"),
            count_Searches=("is_searched", "sum"),
        )
        .reset_index()
    )
    df_returned["month"], df_returned["year"] = (
        df_returned["date"].dt.month_name(),
        df_returned["date"].dt.year,
    )
    df_returned["days"] = np.where(
        df_returned["month"].isin(
            ["January", "March", "May", "July", "August", "October", "December"]
        ),
        31,
        30,
    )
    df_returned["days"] = np.where(
        df_returned["month"] == "February", 28, df_returned["days"]
    )
    df_returned["days"] = np.where(
        (df_returned["month"] == "February") & (df_returned["year"] == 2020),
        29,
        df_returned["days"],
    )
    df_returned["avg_daily_Stops"] = df_returned["count_Stops"] / df_returned["days"]
    df_returned["avg_daily_Searches"] = (
        df_returned["count_Searches"] / df_returned["days"]
    )
    df_returned["percent_change_Stops"] = (
        df_returned["avg_daily_Stops"] - df_returned["avg_daily_Stops"].shift(1)
    ) / df_returned["avg_daily_Stops"].shift(1)
    df_returned["percent_change_Searches"] = (
        df_returned["avg_daily_Searches"] - df_returned["avg_daily_Searches"].shift(1)
    ) / df_returned["avg_daily_Searches"].shift(1)

    df_returned = pd.wide_to_long(
        df_returned,
        stubnames=["count", "avg_daily", "percent_change"],
        i="date",
        j="type",
        sep="_",
        suffix="(!?Stops|Searches)",
    ).reset_index()

    return df_returned

# Calculate overall search percent
df_temp = monthly_stops_and_searches(df_)
TOTAL_STOPS = df_temp.loc[df_temp["type"] == "Stops"]["count"].sum()
TOTAL_SEARCHES = df_temp.loc[df_temp["type"] == "Searches"]["count"].sum()
OVERALL_SEARCH_RATIO = TOTAL_SEARCHES / TOTAL_STOPS * 100

# Create df for average daily stops and searches by alignment grouping
df_temp1 = monthly_stops_and_searches(df_.loc[df_["department"] == "SPPD"])
df_temp1["department"] = "SPPD"
df_temp2 = monthly_stops_and_searches(df_.loc[df_["department"] == "Other Aligned PDs"])
df_temp2["department"] = "Other Aligned PDs"
df_temp3 = monthly_stops_and_searches(
    df_.loc[df_["department"] == "Unchanged Policy PDs"]
)
df_temp3["department"] = "Unchanged Policy PDs"

# Combine data for chart creation
df_chart_stop_search = pd.concat([df_temp1, df_temp2, df_temp3], axis=0)

# Calculate percent of stops attributed to SPPD
df_chart_stop_search.loc[
    (df_chart_stop_search["department"] == "SPPD")
    & (df_chart_stop_search["type"] == "Stops")
]["count"].sum() / TOTAL_STOPS * 100

# Calculate percent of stops for each PD
df_["vehicle_jurisdiction"].value_counts(normalize=True)

df_chart_annual = df_chart_stop_search.copy()

# Create 4 annual periods from Sept-Aug
df_chart_annual = df_chart_annual.loc[
    (df_chart_annual["date"] >= "2018-9-01") & (df_chart_annual["date"] < "2022-9-01")
]

conditions = [
    (df_chart_annual["date"] >= "2018-9-01") & (df_chart_annual["date"] < "2019-9-01"),
    (df_chart_annual["date"] >= "2019-9-01") & (df_chart_annual["date"] < "2020-9-01"),
    (df_chart_annual["date"] >= "2020-9-01") & (df_chart_annual["date"] < "2021-9-01"),
    (df_chart_annual["date"] >= "2021-9-01") & (df_chart_annual["date"] < "2022-9-01"),
]

labels = ["2019", "2020", "2021", "2022"]

df_chart_annual["Time Period"] = np.select(conditions, labels)

df_chart_annual = (
    df_chart_annual.groupby(["type", "department", "Time Period"])
    .agg(mean=("count", "mean"), median=("count", "median"), total=("count", "sum"))
    .reset_index()
)

# Calculate percent change from prior year  
df_chart_annual["percent_mean_change"] = np.where(
    df_chart_annual["Time Period"] == "2019", 
    0, 
    (df_chart_annual["mean"] - df_chart_annual["mean"].shift(1)) / df_chart_annual["mean"].shift(1)
)

df_chart_annual["percent_median_change"] = np.where(
    df_chart_annual["Time Period"] == "2019",
    0,
    (df_chart_annual["median"] - df_chart_annual["median"].shift(1)) / df_chart_annual["median"].shift(1)
)
 
df_chart_annual["percent_total_change"] = np.where(
    df_chart_annual["Time Period"] == "2019", 
    0, 
    (df_chart_annual["total"] - df_chart_annual["total"].shift(1)) / df_chart_annual["total"].shift(1)
)

# Calculate percentage of stops and searches by reason for stop
def get_percentage_stop_search(df, grouping_col):
    """
    Get the percentage of stops and percentage of searches per month per group specified by the
    categorical variable chosen to group by.
    Parameters:
    - df (DataFrame): DataFrame of monthly stops and searches by a categorical grouping variable.
    - grouping_col (str): string of a column name.

    Returns:
    - df_returned (DataFrame): DataFrame aggregated to reflect the monthly number of stops, number of
    searches, and percentage of stops and searches per subgroup specified by the grouping_col.
    """
    df_returned = df.copy()
    df_returned["total_monthly_Stops"] = df_returned.groupby("date")[
        "monthly_Stops"
    ].transform("sum")
    df_returned["total_monthly_Searches"] = df_returned.groupby("date")[
        "monthly_Searches"
    ].transform("sum")
    df_returned["percentage_Stops"] = df_returned["monthly_Stops"].div(
        df_returned["total_monthly_Stops"]
    )
    df_returned["percentage_Searches"] = df_returned["monthly_Searches"].div(
        df_returned["total_monthly_Searches"]
    )
    df_returned = pd.wide_to_long(
        df_returned,
        stubnames=["monthly", "total_monthly", "percentage"],
        i=["date", grouping_col],
        j="type",
        sep="_",
        suffix="(!?Stops|Searches)",
    ).reset_index()

    return df_returned


# Create new df for traffic stop reason
df_stop_reason = df_.copy()

# Group into 1 year before, 1 year after policy
df_stop_reason = df_stop_reason.loc[
    (df_stop_reason["date"] >= "2020-9-01") & (df_stop_reason["date"] < "2022-9-01")
]

conditions = [
    (df_stop_reason["date"] >= "2020-9-01") & (df_stop_reason["date"] < "2021-9-01"),
    (df_stop_reason["date"] >= "2021-9-01") & (df_stop_reason["date"] < "2022-9-01"),
]

labels = ["Before", "After"]

df_stop_reason["Time Period"] = np.select(conditions, labels)

# Include top 3 reasons, along with everything else grouped into "other"
df_stop_reason["reason"] = df_stop_reason["reason_for_stop_1"].copy()
df_stop_reason["reason"] = np.where(
    df_stop_reason["reason_for_stop_1"].isin(["Citizen Report", "No Reason Given"]),
    "Other",
    df_stop_reason["reason"],
)

df_stop_reason = (
    df_stop_reason.groupby(["department", "Time Period", "reason"])
    .agg(
        reas_tot_stops=("master_incident_number", "nunique"),
        reas_tot_searches=("is_searched", "sum"),
    )
    .reset_index()
)

df_stop_reason["total_stops"] = df_stop_reason.groupby(["department", "Time Period"])[
    "reas_tot_stops"
].transform("sum")
df_stop_reason["perc_stops"] = (
    df_stop_reason["reas_tot_stops"].div(df_stop_reason["total_stops"]) * 100
).round(1)

df_stop_reason["total_searches"] = df_stop_reason.groupby(
    ["department", "Time Period"]
)["reas_tot_searches"].transform("sum")
df_stop_reason["perc_searches"] = (
    df_stop_reason["reas_tot_searches"].div(df_stop_reason["total_searches"]) * 100
).round(1)

# Calculate percent change
def percent_change(df, column_name, new_column):
    """
    Calculate the percentage change from the previous period.
    Parameters:
    - df (DataFrame): DataFrame
    - column_name (str): name of the column with the numeric value that should be used to calculate
    the percent change
    - new_column (str): name of the new column to create

    Returns:
    - df (DataFrame): DataFrame
    """
    df[new_column] = np.where(
        df["Time Period"] == "After",
        (df[column_name] - df[column_name].shift(-1)) / df[column_name].shift(-1),
        0,
    )

    return df


# Create df for percent change in moving and vehicle equipment violations
df_moving = df_stop_reason.loc[df_stop_reason["reason"] == "Moving Violation"].copy()
df_moving = percent_change(df_moving, "reas_tot_stops", "percent_change_stops")
df_moving = percent_change(df_moving, "reas_tot_searches", "percent_change_searches")

df_vehicle = df_stop_reason.loc[df_stop_reason["reason"] == "Vehicle Violation"].copy()
df_vehicle = percent_change(df_vehicle, "reas_tot_stops", "percent_change_stops")
df_vehicle = percent_change(df_vehicle, "reas_tot_searches", "percent_change_searches")

df_reason = pd.concat([df_moving, df_vehicle], axis=0)
df_reason = pd.wide_to_long(
    df_reason,
    stubnames=["reas_tot", "total", "perc", "percent_change"],
    i=["department", "Time Period", "reason"],
    j="type",
    sep="_",
    suffix="(!?stops|searches)",
).reset_index()
df_reason["type"] = df_reason["type"].replace(
    {"stops": "Stops", "searches": "Searches"}
)

"""
Driver race and ethnicity statistics.
Per capita numbers were gathered from https://www.census.gov/quickfacts/ramseycountyminnesota
"""
# Moving violations

df_temp1 = get_percentage_stop_search(
    get_monthly_stops_searches(
        df_.loc[df_["reason_for_stop_1"] == "Moving Violation"], "race_1"
    ),
    "race_1",
)
df_temp1["reason"] = "Moving Violations"

df_temp2 = get_percentage_stop_search(
    get_monthly_stops_searches(
        df_.loc[df_["reason_for_stop_1"] == "Vehicle Violation"], "race_1"
    ),
    "race_1",
)
df_temp2["reason"] = "Vehicle Equipment Violations"

df_race_reason = pd.concat([df_temp1, df_temp2], axis=0)

df_race_reason = df_race_reason.loc[
    (df_race_reason["date"] >= "2018-9-01") & (df_race_reason["date"] < "2022-9-01")
]

conditions = [
    (df_race_reason["date"] >= "2018-9-01") & (df_race_reason["date"] < "2019-9-01"),
    (df_race_reason["date"] >= "2019-9-01") & (df_race_reason["date"] < "2020-9-01"),
    (df_race_reason["date"] >= "2020-9-01") & (df_race_reason["date"] < "2021-9-01"),
    (df_race_reason["date"] >= "2021-9-01") & (df_race_reason["date"] < "2022-9-01"),
]

labels = ["9/18-8/19", "9/19-8/20", "9/20-8/21", "9/21-8/22"]

df_race_reason["Time Period"] = np.select(conditions, labels)

df_race_reason = (
    df_race_reason.groupby(["type", "reason", "race_1", "Time Period"])
    .agg(
        mean_monthly=("monthly", "mean"),
        median_monthly=("monthly", "median"),
        total_annual=("monthly", "sum"),
    )
    .reset_index()
)

df_race_reason["percent_mean_change"] = np.where(
    df_race_reason["Time Period"] == "9/18-9/19",
    0,
    (df_race_reason["mean_monthly"] - df_race_reason["mean_monthly"].shift(1)) / df_race_reason["mean_monthly"].shift(1)
)

df_race_reason["percent_median_change"] = np.where(
    df_race_reason["Time Period"] == "9/18-9/19",
    0,
    (df_race_reason["median_monthly"] - df_race_reason["median_monthly"].shift(1)) / df_race_reason["median_monthly"].shift(1)
)

RC_TOTAL_POP = 543257

df_race_reason["pop_percent"] = df_race_reason["race_1"].copy()
df_race_reason["pop_percent"] = df_race_reason["pop_percent"].replace(
    {
        "Asian": 0.157,
        "Black": 0.134,
        "Hispanic or Latino": 0.077,
        "Native American": 0.010,
        "Other": 0.039,
        "White": 0.599,
    }
)

df_race_reason["pop_percent"] = pd.to_numeric(df_race_reason["pop_percent"])

df_race_reason["pop_count"] = df_race_reason["pop_percent"] * RC_TOTAL_POP
df_race_reason["count_per_cap"] = (
    df_race_reason["total_annual"] / df_race_reason["pop_count"]
) * 1000
 
df_race_reason["percent_per_cap_change"] = np.where(
    df_race_reason["Time Period"] == "9/18-9/19",
    0,
    (df_race_reason["count_per_cap"] - df_race_reason["count_per_cap"].shift(1)) / df_race_reason["count_per_cap"].shift(1)
)

conditions = [
    (df_race_reason["Time Period"] == "9/20-8/21"),
    (df_race_reason["Time Period"] == "9/21-8/22"),
]

labels = ["Before", "After"]

df_race_reason["period"] = np.select(conditions, labels)

df_chart = df_race_reason.loc[
    (df_race_reason["type"] == "Stops") & (df_race_reason["race_1"] != "Other")
]

df_chart = df_race_reason.loc[
    (df_race_reason["type"] == "Searches") & (df_race_reason["race_1"] != "Other")
]

df_chart = df_race_reason.loc[
    (df_race_reason["type"] == "Stops")
    & (df_race_reason["race_1"] != "Other")
    & (df_race_reason["period"] == "After")
]

df_chart = df_race_reason.loc[
    (df_race_reason["type"] == "Searches")
    & (df_race_reason["race_1"] != "Other")
    & (df_race_reason["period"] == "After")
]

########################################################################
# 911 calls analysis
########################################################################
df_calls = pd.read_parquet("code_f/data/all_911_calls.parquet")

# Fill na values
df_calls["problem"] = df_calls["problem"].fillna(value="None")

# Remove test calls
df_calls = df_calls.loc[(df_calls["problem"] != "TEST CALL")]

# subtract 7 days from date, strip timezone, and filter to post-2015
df_calls["date"] = pd.to_datetime(df_calls["time_callenteredqueue"]) - pd.Timedelta(
    days=7
)

df_calls = df_calls.loc[
    pd.to_datetime(df_calls["date"]) >= pd.to_datetime("2016-01-01")
]
df_calls = df_calls.loc[pd.to_datetime(df_calls["date"]) < pd.to_datetime("2022-10-01")]

"""
Filtering data
We filter data in the steps below. Included are print statements that we
use as a reference to check the assumptions we are making. Robustness
checks on the analysis whereby these steps were removed were done for most
filters.
"""
# Calls without a master_incident_number were previously removed at the direction of ECC
total_calls = len(df_calls)
df_calls = df_calls.loc[df_calls.master_incident_number.notna()]
print(
    "Number of rows after removing calls missing master incident numbers",
    str(len(df_calls)),
)

print(
    "Number of rows removed for missing master incident number",
    str(total_calls - len(df_calls)),
)
total_calls = len(df_calls)

df_calls["call_disposition"] = df_calls["call_disposition"].fillna(value="None")
df_calls_no_cancels = df_calls.loc[
    ~(df_calls.call_disposition.str.contains("cance", case=False))
   ]
print("Number of rows after removing canceled calls", str(len(df_calls_no_cancels)))
print(
    "Number of rows removed for canceled", str(total_calls - len(df_calls_no_cancels))
)
total_calls = len(df_calls_no_cancels)

df_calls_no_cancels_hangups = df_calls_no_cancels.loc[
    ~(df_calls_no_cancels.call_disposition.str.contains("hang", case=False))
]

print(
    "Number of rows after removing hang up calls", str(len(df_calls_no_cancels_hangups))
)
print(
    "Number of rows removed for hangups",
    str(total_calls - len(df_calls_no_cancels_hangups)),
)
total_calls = len(df_calls_no_cancels_hangups)

df_calls_no_cancels_hangups_missing_dispo = df_calls_no_cancels_hangups.loc[
    df_calls_no_cancels_hangups["call_disposition"] != "None"
]

print(
    "Number of rows after removing missing call dispositions",
    str(len(df_calls_no_cancels_hangups_missing_dispo)),
)
print(
    "Number of rows removed for missing a call disposition",
    str(total_calls - len(df_calls_no_cancels_hangups_missing_dispo)),
)
total_calls = len(df_calls_no_cancels_hangups_missing_dispo)

# Drop duplicates

important_calls_cols = [
    "id",
    "master_incident_number",
    "response_date",
    "jurisdiction",
    "incident_type",
    "problem",
    "priority_number",
    "priority_description",
    "call_disposition",
    "cancel_reason",
    "geoid10",
    "incident_jurisdiction",
    "policy",
]
df_calls_cleaned_up = df_calls_no_cancels_hangups_missing_dispo.drop(
    columns="index"
).drop_duplicates(subset=important_calls_cols)

print("Number of rows after removing duplicate rows", str(len(df_calls_cleaned_up)))
print(
    "Number of rows removed for being a duplicate",
    str(total_calls - len(df_calls_cleaned_up)),
)
total_calls = len(df_calls_cleaned_up)

print("Number of rows currently", str(len(df_calls_cleaned_up)))
print(
    "Number of rows to be dropped by deduplicating on master_incident_number",
    str(total_calls - df_calls_cleaned_up["master_incident_number"].nunique()),
)
print(
    "Number of rows with missing master_incident_number",
    str(len(df_calls_cleaned_up[df_calls_cleaned_up["master_incident_number"].isna()])),
)

df_calls = df_calls_cleaned_up.copy()

"""
Label calls with the PD that responds most frequently to that census tract.
Rather than using the department that responded to a given call, we are relabeling
a row by the jx that most frequently responds to calls in that census tract. This
better estimates the calls arising out of a given jx territory which better reflects
the impact of a jx adhering to the policy. To do this, we restrict the sample we
look at to the year before the policy, this avoids the impact of any changes to
census tracts that occurred before 2020.
"""
# Extract first 11 digits from geoid
df_calls["geoid_tract"] = df_calls["geoid10"].str.extract(r"(\d{11})")

# Restrict to 1 year before policy
df_calls_1yr_prepolicy = df_calls[
    (pd.to_datetime(df_calls["date"]) >= (POLICY_START_DATE - pd.DateOffset(years=1)))
    & (pd.to_datetime(df_calls["date"]) < POLICY_START_DATE)
]

# Get total calls by tract-jurisdiction
calls_by_tract_juris = (
    df_calls_1yr_prepolicy.groupby(["geoid_tract", "jurisdiction"])
    .size()
    .to_frame(name="count")
    .reset_index()
)

# Get total calls by tract
calls_by_tract = (
    df_calls_1yr_prepolicy.groupby(["geoid_tract"])
    .size()
    .to_frame(name="total_count")
    .reset_index()
)

# Merge by tract
df_labels = pd.merge(
    calls_by_tract_juris, calls_by_tract, on="geoid_tract", validate="m:1"
)

# Create share of calls responded by each PD
df_labels["share"] = df_labels["count"] / df_labels["total_count"]

# Keep the PD with the highest response rate in tract
df_labels = (
    df_labels.sort_values(
        by="share",
        ascending=True,
    )
    .drop_duplicates(["geoid_tract"], keep="last")
    .reset_index(drop=True)
    .rename(
        columns={
            "jurisdiction": "jurisdiction_max_response",
            "count": "calls_by_jurisdiction",
            "total_count": "total_calls_made_from_tract",
            "share": "share_of_calls_by_jurisdiction",
        },
    )
)

df_calls_merged = df_calls.merge(
    df_labels, on="geoid_tract", how="left", validate="m:1"
)

# Checking what the values of the jurisdiction counts are
df_calls_merged["jurisdiction_max_response"].value_counts()
df_calls = df_calls_merged.copy()

# Categorize by policy alignment by expected police response department
# Reset policy variable based on jurisdiction_max_response
df_calls["policy"] = "Other Aligned PDs"
df_calls["policy"].mask(
    df_calls["jurisdiction_max_response"].isin(["SPPD"]), "SPPD", inplace=True
)
df_calls["policy"].mask(
    df_calls["jurisdiction_max_response"].isin(
        ["RCSO", "WBPD", "MVPD", "NSPD", "NBPD"]
    ),
    "Unchanged Policy PDs",
    inplace=True,
)
df_calls["policy"].mask(
    df_calls["jurisdiction_max_response"].isin(["FGPD"]), "Unknown", inplace=True
)
df_calls["policy"].mask(
    (df_calls["jurisdiction_max_response"].isna()), "missing", inplace=True
)

# Categorize call types by problem type, then group call categories

conditions_crime = [
    (df_calls["problem"].isin(["911 - Investigate 911 Hangup"])),
    (
        df_calls["problem"].isin(
            [
                "BRG - Burglary",
                "CDP - Criminal Damage Property",
                "MVT - Motor Vehicle Theft",
                "SHP - Shoplifter",
                "THA - Theft From Auto",
                "THF - Theft",
                "APD - Accident Property Damage",
                "FRD - Fraud or Forgery",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "TRF - Traffic Stop",
                "TSI - Traffic Safety Init",
                "AHR - Accident Hit and Run",
                "DKD - Drunk Driver",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "BOM - Bomb Threat",
                "HRS - Harassment Report",
                "DOC - Disorderly Conduct",
                "JUV - Juvenile Incidents",
                "POR - Predatory Offender Reg",
                "PROB - Probation Violation",
                "VIC - Vice Prostitution",
                "VOP - Prot Order Violation",
                "ALA - Alarm Sounding",
                "ALP - Alarm Panic/Holdup/Dures",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "CIV - Civil Problem",
                "DIS - Dispute Disagreement",
                "LOC - Lockout",
                "FGT - Fight",
            ]
        )
    ),
    (df_calls["problem"].isin(["DOM -Domestic Fam Relationship", "CAB - Child Abuse"])),
    (
        df_calls["problem"].isin(
            [
                "ASS - Assault",
                "CSC - Crim Sexual Conduct",
                "HRI - Hit & Run Acc w/Injuries",
                "KDN - Kidnapping - Amber Alert",
                "ROB - Robbery",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "AFA - Assist Fire Agency",
                "AMA - Assist Medical Agency",
                "AOA - Assist Other Agency",
                "PIC - Person in Crisis",
                "EDP - Emotionally Disturb Pers",
                "AST - Assist Citizen",
                "PLS - Person Location Search",
                "PLS - Project Lifesaver",
                "SIP - Suicide In Progress",
                "WTR - Water Patrol Response",
                "DKP- Drunk Person",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "ADM - Administrative Detail",
                "ANC - Animal Complaint",
                "COE - Community Outreach Event",
                "ATL - Attempt To Locate",
                "AWI - Accident With Injuries",
                "PLR - Prowler",
                "BAIT - Bait Car Activation",
                "CHK - Records Check",
                "COT - Contracted Overtime",
                "DEM - Demonstration/Protest",
                "MSP - Missing Person, Juvenile",
                "PPV - Police Proactive Visit",
                "MTG - Meeting Officer Assigned",
                "ODE - Off Duty Employment",
                "PFP - Proactive Foot Patrols",
                "LOS - Lost Property",
                "RPR - Recovered Property",
                "SOL - Solicitor",
                "SUS - Suspicious Activity",
                "RCIV - Civil Div Paper Serve",
                "TOW - Tow",
                "TRP - Transport",
                "UOF - Use of Force",
                "ESC - Escape From Custody",
                "DOA - Death Investigation",
                "PCN - Previous Case Follow-Up",
                "WAR - Warrant",
                "WPN - Weapon",
                "PRP - Problem Property",
                "DGC - Dangerous Condition",
                "CMT - County Maintenance",
                "BTOW - Tow For Ord Violation",
                "ABV - Abandoned Vehicle",
                "PRK - Parking Complaint",
                "WEL - Welfare Check",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            [
                "COD - Code Enforcement",
                "DSB - Disturbance Noise Comp",
                "FWK - Fireworks",
                "GRF - Graffiti",
            ]
        )
    ),
    (
        df_calls["problem"].isin(
            ["ARS - Arson Investigation", "INV - Investigate", "SHF - Shots Fired"]
        )
    ),
    (df_calls["problem"].isin(["NAR - Narcotics"])),
]

labels_crime = [
    "911 hangup",
    "Property",
    "Vehicle",
    "Other Crime",
    "Civil Problem",
    "Domestic Violence/Disturbance",
    "Crime involving injury/violence",
    "Support - Person, Alarm, or Agency",
    "Other",
    "Civil Ordinance Issue",
    "Investigatory",
    "Drugs",
]

df_calls["general_problem"] = np.select(conditions_crime, labels_crime)

conditions_crime2 = [
    (
        df_calls["general_problem"].isin(
            ["Property", "Drugs", "Weapons", "Other Crime", "Vehicle"]
        )
    ),
    (
        df_calls["general_problem"].isin(
            ["Crime involving injury/violence", "Domestic Violence/Disturbance"]
        )
    ),
    (df_calls["general_problem"] == "Investigatory"),
    (df_calls["general_problem"] == "Civil Ordinance Issue"),
]

labels_crime2 = ["Non-Violent", "Violent", "Investigatory", "Civil Ordinance"]

df_calls["crime"] = np.select(conditions_crime2, labels_crime2)
df_calls["crime"] = df_calls["crime"].replace({"0": "Other"})

# Filter to groups of interest
df_calls = df_calls.loc[
    df_calls["policy"].isin(["SPPD", "Other Aligned PDs", "Unchanged Policy PDs"])
]

# Filter to dates of interest
df_calls = df_calls.loc[
    (df_calls["date"] >= ALL_PDS_START_DATE) & (df_calls["date"] <= ALL_PDS_END_DATE)
]

# Calculate percent change in average daily calls
def monthly_calls(df):
    """
    Calculate the average daily number of 911 calls after accounting for the number of days in each
    month.
    Parameters:
    - df (DataFrame): DataFrame of all 911 calls.

    Returns:
    - df_returned (DataFrame): DataFrame grouped by year-month and providing the average number of
    daily 911 calls.
    """
    df_returned = (
        df.groupby([pd.Grouper(key="date", freq="MS")])
        .agg(Calls=("master_incident_number", "nunique"))
        .reset_index()
    )
    df_returned["month"], df_returned["year"] = (
        df_returned["date"].dt.month_name(),
        df_returned["date"].dt.year,
    )
    df_returned["days"] = np.where(
        df_returned["month"].isin(
            ["January", "March", "May", "July", "August", "October", "December"]
        ),
        31,
        30,
    )
    df_returned["days"] = np.where(
        df_returned["month"] == "February", 28, df_returned["days"]
    )
    df_returned["days"] = np.where(
        (df_returned["month"] == "February") & (df_returned["year"] == 2020),
        29,
        df_returned["days"],
    )
    df_returned["avg_daily_Calls"] = df_returned["Calls"] / df_returned["days"]
    df_returned["percent_change_Calls"] = (
        df_returned["avg_daily_Calls"] - df_returned["avg_daily_Calls"].shift(1)
    ) / df_returned["avg_daily_Calls"].shift(1)

    return df_returned


# Calculate percent change in average daily calls - with 1 grouping variable
def monthly_calls_by_category(df, agg_col):
    """
    Calculate the average daily number of 911 calls after accounting for the number of days in each
    month and by grouping by a categorical variable.
    Parameters:
    - df (DataFrame): DataFrame of all 911 calls.
    - agg_col (str): name of categorical column to group the dataframe by.

    Returns:
    - df_returned (DataFrame): DataFrame grouped by year-month and providing the average number of
    daily 911 calls.
    """
    df_returned = (
        df.groupby([agg_col, pd.Grouper(key="date", freq="MS")])
        .agg(Calls=("master_incident_number", "nunique"))
        .reset_index()
    )
    df_returned["month"], df_returned["year"] = (
        df_returned["date"].dt.month_name(),
        df_returned["date"].dt.year,
    )
    df_returned["days"] = np.where(
        df_returned["month"].isin(
            ["January", "March", "May", "July", "August", "October", "December"]
        ),
        31,
        30,
    )
    df_returned["days"] = np.where(
        df_returned["month"] == "February", 28, df_returned["days"]
    )
    df_returned["days"] = np.where(
        (df_returned["month"] == "February") & (df_returned["year"] == 2020),
        29,
        df_returned["days"],
    )
    df_returned["avg_daily_Calls"] = df_returned["Calls"] / df_returned["days"]
    df_returned["percent_change_Calls"] = (
        df_returned["avg_daily_Calls"] - df_returned["avg_daily_Calls"].shift(1)
    ) / df_returned["avg_daily_Calls"].shift(1)

    return df_returned

df_chart_calls = monthly_calls_by_category(df_calls, "policy")

# Look at trends by crime type broadly
df_temp1 = monthly_calls_by_category(
    df_calls.loc[df_calls["policy"] == "SPPD"], "crime"
)
df_temp1["department"] = "SPPD"

df_temp2 = monthly_calls_by_category(
    df_calls.loc[df_calls["policy"] == "Other Aligned PDs"], "crime"
)
df_temp2["department"] = "Other Aligned PDs"

df_temp3 = monthly_calls_by_category(
    df_calls.loc[df_calls["policy"] == "Unchanged Policy PDs"], "crime"
)
df_temp3["department"] = "Unchanged Policy PDs"

df_calls_crime = pd.concat([df_temp1, df_temp2, df_temp3], axis=0)

# Create new df for percent change
df_chart_crime = df_calls_crime.copy()

# Group into 1 year before, 1 year after policy
df_chart_crime = df_chart_crime.loc[
    (df_chart_crime["date"] >= "2020-9-01") & (df_chart_crime["date"] < "2022-9-01")
]

conditions = [
    (df_chart_crime["date"] >= "2020-9-01") & (df_chart_crime["date"] < "2021-9-01"),
    (df_chart_crime["date"] >= "2021-9-01") & (df_chart_crime["date"] < "2022-9-01"),
]

labels = ["Before", "After"]

df_chart_crime["Time Period"] = np.select(conditions, labels)

df_chart_crime = (
    df_chart_crime.groupby(["department", "crime", "Time Period"])
    .agg(crime_calls=("Calls", "sum"))
    .reset_index()
)

# Calculate percent change in violent crime
df_violent = df_chart_crime.loc[df_chart_crime["crime"] == "Violent"].copy()
df_violent["percent_change"] = np.where(
    df_violent["Time Period"] == "Before", 
    0, 
    (df_violent["crime_calls"] - df_violent["crime_calls"].shift(-1)) / df_violent["crime_calls"].shift(-1)
)

# Calculate percent change in non-violent crime
df_nonviolent = df_chart_crime.loc[df_chart_crime["crime"] == "Non-Violent"].copy()
df_nonviolent["percent_change"] = 
df_nonviolent["percent_change"] = np.where(
    df_nonviolent["Time Period"] == "Before", 
    0, 
    (df_nonviolent["crime_calls"] - df_nonviolent["crime_calls"].shift(-1)) / df_nonviolent["crime_calls"].shift(-1)
)

df_chart_violence = pd.concat([df_violent, df_nonviolent], axis=0)

# Look at contraband-related crimes
df_calls_crime = monthly_calls_by_category(df_calls, "crime")

Contraband_Cats = ["WPN - Weapon", "SHF - Shots Fired", "NAR - Narcotics"]

df_contraband = df_calls.loc[df_calls["problem"].isin(Contraband_Cats)]
df_contraband = monthly_calls_by_category(df_contraband, "problem")
df_contraband["problem"] = df_contraband["problem"].replace(
    {
        "WPN - Weapon": "Weapon",
        "SHF - Shots Fired": "Shots Fired",
        "NAR - Narcotics": "Narcotics",
    }
)

"""
JIL received an additional data source of gun seizures by SPPD. That dataset
covered a limited time period and is relatively small thus cleaning, manipulation,
and analysis were all done in the code below.
"""
df_firearms = pd.read_excel("code_f/data/firearm_recovery.xlsx")

# Subtract 7 seven days from date to match other analyses
df_firearms["Date"] = df_firearms["Date"] - pd.Timedelta(days=7)

# Fix data entry
df_firearms["Recovered on Traffic Stop?"] = df_firearms[
    "Recovered on Traffic Stop?"
].replace({"yes": "Yes"})

# All seizures
df_chart_all_seizures = (
    df_firearms.groupby([pd.Grouper(key="Date", freq="MS")])
    .agg(seizures=("Case Number", "count"))
    .reset_index()
)
df_chart_all_seizures["month"], df_chart_all_seizures["year"] = (
    df_chart_all_seizures["Date"].dt.month_name(),
    df_chart_all_seizures["Date"].dt.year,
)
df_chart_all_seizures["days"] = np.where(
    df_chart_all_seizures["month"].isin(
        ["January", "March", "May", "July", "August", "October", "December"]
    ),
    31,
    30,
)
df_chart_all_seizures["days"] = np.where(
    df_chart_all_seizures["month"] == "February", 28, df_chart_all_seizures["days"]
)
df_chart_all_seizures["days"] = np.where(
    (df_chart_all_seizures["month"] == "February")
    & (df_chart_all_seizures["year"] == 2020),
    29,
    df_chart_all_seizures["days"],
)
df_chart_all_seizures["avg_daily_seizures"] = (
    df_chart_all_seizures["seizures"] / df_chart_all_seizures["days"]
)
df_chart_all_seizures["cat"] = "All Firearm Seizures"

# Remove tracked data prior to 2020 that is due to the date shift and partial month data from October 2022
df_chart_all_seizures = df_chart_all_seizures.loc[
    ~df_chart_all_seizures["Date"].isin(["2019-12-01", "2022-10-01"])
]

# Traffic seizures
df_chart_traffic_seizures = (
    df_firearms.loc[df_firearms["Recovered on Traffic Stop?"] == "Yes"]
    .groupby([pd.Grouper(key="Date", freq="MS")])
    .agg(seizures=("Case Number", "count"))
    .reset_index()
)
df_chart_traffic_seizures["month"], df_chart_traffic_seizures["year"] = (
    df_chart_traffic_seizures["Date"].dt.month_name(),
    df_chart_traffic_seizures["Date"].dt.year,
)
df_chart_traffic_seizures["days"] = np.where(
    df_chart_traffic_seizures["month"].isin(
        ["January", "March", "May", "July", "August", "October", "December"]
    ),
    31,
    30,
)
df_chart_traffic_seizures["days"] = np.where(
    df_chart_traffic_seizures["month"] == "February",
    28,
    df_chart_traffic_seizures["days"],
)
df_chart_traffic_seizures["days"] = np.where(
    (df_chart_traffic_seizures["month"] == "February")
    & (df_chart_traffic_seizures["year"] == 2020),
    29,
    df_chart_traffic_seizures["days"],
)
df_chart_traffic_seizures["avg_daily_seizures"] = (
    df_chart_traffic_seizures["seizures"] / df_chart_traffic_seizures["days"]
)
df_chart_traffic_seizures["cat"] = "Traffic Stop Firearm Seizures"

# Remove tracked data prior to 2020 that is due to the date shift and partial month data from October 2022
df_chart_traffic_seizures = df_chart_traffic_seizures.loc[
    ~df_chart_traffic_seizures["Date"].isin(["2020-12-01", "2022-10-01"])
]

# Combine chart dfs
df_chart_guns = pd.concat([df_chart_all_seizures, df_chart_traffic_seizures], axis=0)

# Add new variable for before and after policy implementation
conditions = [
    (df_chart_guns["Date"] < "2021-9-01"),
    (df_chart_guns["Date"] >= "2021-9-01"),
]

labels = ["Before", "After"]

df_chart_guns["Time Period"] = np.select(conditions, labels)

# Firearm seizure statistics - all seizures
MEAN_SEIZ_BEFORE = df_chart_guns.loc[
    (df_chart_guns["Time Period"] == "Before")
    & (df_chart_guns["cat"] == "All Firearm Seizures")
]["avg_daily_seizures"].mean()
MEAN_SEIZ_AFTER = df_chart_guns.loc[
    (df_chart_guns["Time Period"] == "After")
    & (df_chart_guns["cat"] == "All Firearm Seizures")
]["avg_daily_seizures"].mean()
MEAN_SEIZ_OVERALL = df_chart_guns.loc[df_chart_guns["cat"] == "All Firearm Seizures"][
    "avg_daily_seizures"
].mean()

# Firearm seizure statistics - traffic seizures
MEAN_TRAF_BEFORE = df_chart_guns.loc[
    (df_chart_guns["Time Period"] == "Before")
    & (df_chart_guns["cat"] == "Traffic Stop Firearm Seizures")
]["avg_daily_seizures"].mean()
MEAN_TRAF_AFTER = df_chart_guns.loc[
    (df_chart_guns["Time Period"] == "After")
    & (df_chart_guns["cat"] == "Traffic Stop Firearm Seizures")
]["avg_daily_seizures"].mean()
MEAN_TRAF_OVERALL = df_chart_guns.loc[
    df_chart_guns["cat"] == "Traffic Stop Firearm Seizures"
]["avg_daily_seizures"].mean()

print(
    f"The SPPD seized {MEAN_SEIZ_BEFORE:.1f} firearms a day on average per month prior to policy implementation, {MEAN_SEIZ_AFTER:.1f} after policy implementation, and {MEAN_SEIZ_OVERALL:.1f} overall."
)

print(
    f"The SPPD seized {MEAN_TRAF_BEFORE:.1f} firearms a day during traffic stops on average per month prior to policy implementation, {MEAN_TRAF_AFTER:.1f} after policy implementation, and {MEAN_TRAF_OVERALL:.1f} overall."
)

# Traffic stop gun seizure rate

# SPPD stops and searches, Jan 2021 - Sept 2022
df_sppd_stops = df_chart_stop_search.loc[df_chart_stop_search["department"] == "SPPD"]
df_sppd_stops = df_sppd_stops.loc[
    (df_sppd_stops["date"] >= "2021-01-01") & (df_sppd_stops["date"] <= "2022-09-01")
]
df_sppd_stops = df_sppd_stops[["date", "type", "count"]]

# SPPD traffic stop gun seizures, Jan 2021 - Sept 2022
df_sppd_seizures = df_chart_guns[["Date", "seizures", "cat"]]
df_sppd_seizures = df_sppd_seizures.loc[
    df_sppd_seizures["cat"] == "Traffic Stop Firearm Seizures"
]
df_sppd_seizures.rename(
    columns={"Date": "date", "seizures": "count", "cat": "type"}, inplace=True
)
df_sppd_seizures["type"] = df_sppd_seizures["type"].replace(
    {"Traffic Stop Firearm Seizures": "Seizures"}
)

# Combine data
df_chart_sppd_seizures = pd.concat([df_sppd_stops, df_sppd_seizures], axis=0)
df_chart_sppd_seizures = pd.pivot(
    df_chart_sppd_seizures, index=["date"], columns=["type"], values="count"
).reset_index()

# Add seizure rate columns
df_chart_sppd_seizures["seizures_per_stop"] = (
    df_chart_sppd_seizures["Seizures"] / df_chart_sppd_seizures["Stops"]
)
df_chart_sppd_seizures["seizures_per_search"] = (
    df_chart_sppd_seizures["Seizures"] / df_chart_sppd_seizures["Searches"]
)

df_chart_seizure_rate = df_chart_sppd_seizures[
    ["date", "seizures_per_stop", "seizures_per_search"]
]
df_chart_seizure_rate = pd.melt(
    df_chart_seizure_rate,
    id_vars="date",
    value_vars=["seizures_per_stop", "seizures_per_search"],
    var_name="type",
    value_name="seizure_rate",
)

# Add new variable for before and after policy implementation
conditions = [
    (df_chart_seizure_rate["date"] < "2021-9-01"),
    (df_chart_seizure_rate["date"] >= "2021-9-01"),
]

labels = ["Before", "After"]

df_chart_seizure_rate["Time Period"] = np.select(conditions, labels)

# Gun seizure statistics - stops
MEAN_SEIZ_STOPS_BEFORE = (
    df_chart_seizure_rate.loc[
        (df_chart_seizure_rate["Time Period"] == "Before")
        & (df_chart_seizure_rate["type"] == "Seizures per Stop")
    ]["seizure_rate"].mean()
    * 100
)
MEAN_SEIZ_STOPS_AFTER = (
    df_chart_seizure_rate.loc[
        (df_chart_seizure_rate["Time Period"] == "After")
        & (df_chart_seizure_rate["type"] == "Seizures per Stop")
    ]["seizure_rate"].mean()
    * 100
)
MEAN_SEIZ_STOPS_OVERALL = (
    df_chart_seizure_rate.loc[df_chart_seizure_rate["type"] == "Seizures per Stop"][
        "seizure_rate"
    ].mean()
    * 100
)

# Gun seizure statistics - searches
MEAN_TRAF_SEARCH_BEFORE = (
    df_chart_seizure_rate.loc[
        (df_chart_seizure_rate["Time Period"] == "Before")
        & (df_chart_seizure_rate["type"] == "Seizures per Search")
    ]["seizure_rate"].mean()
    * 100
)
MEAN_TRAF_SEARCH_AFTER = (
    df_chart_seizure_rate.loc[
        (df_chart_seizure_rate["Time Period"] == "After")
        & (df_chart_seizure_rate["type"] == "Seizures per Search")
    ]["seizure_rate"].mean()
    * 100
)
MEAN_TRAF_SEARCH_OVERALL = (
    df_chart_seizure_rate.loc[df_chart_seizure_rate["type"] == "Seizures per Search"][
        "seizure_rate"
    ].mean()
    * 100
)

print(
    f"The SPPD seized firearms during {MEAN_SEIZ_STOPS_BEFORE:.1f}% of stops on average per month prior to policy implementation, {MEAN_SEIZ_STOPS_AFTER:.1f}% after policy implementation, and {MEAN_SEIZ_STOPS_OVERALL:.1f}% overall."
)

print(
    f"The SPPD seized firearms druing {MEAN_TRAF_SEARCH_BEFORE:.1f}% of searches on average per month prior to policy implementation, {MEAN_TRAF_SEARCH_AFTER:.1f}% after policy implementation, and {MEAN_TRAF_SEARCH_OVERALL:.1f}% overall."
)

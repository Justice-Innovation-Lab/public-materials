# For our analysis, we received two datasets from Ramsey County ECC. One
# contains all calls for service from January 2016 through September 2022.
# The other contains all traffic stop calls for service from January 2016
# through September 2022. Though the traffic stop data is a subset of the
# all calls data, it contains an additional text column with the 'comment'
# from the officer that details the reason for the stop, the race and
# gender of the individual stopped, and the whether the car and person
# were searched.

# The code below details the cleaning and manipulation of both datasets.
# The cleaning and manipulation of the datasets was done in both python
# and R for various steps. This is because our analysis team consisted
# of people more familiar with each respective language and that were
# responsible for different sections of the cleaning and manipulation.
# The cleaning and manipulation was originally done in coding notebooks
# that also contain many smaller, investigative steps. The code presented
# here is a streamlined version of the code to highlight the necessary
# manipulation steps.

# This script picks up from the earlier python code in order to parse
# text comments for demographic information and merge information
# regarding police departments.

library("tidyverse") # all the good stuff
library("magrittr") # Pipes on pipes on pipes
library("arrow") # For dealing with parquet files

###########################################################################
# this section of code deals with the all calls data. It limits the data to
# 911 calls for the crime analysis done later and assigns departments based
# on alignment to the policy
###########################################################################
df_all_calls <- arrow::read_parquet("path_to_data.parquet")

calls_911 <- df_all_calls %>%
    filter(is_911_call=="True" | Caller_Type=="911 Call")

names(calls_911) <- str_to_lower(names(calls_911))

cols <- names(calls_911)
calls_911 <- calls_911 %>%
    mutate(across(all_of(cols), as.character)) %>%
    replace(.=="None" | .=="nan", NA_character_)

# there are many rows that are missing the jurisdiction. For these, we can
# assign the jurisdiction that is most common to the city. we will also use
# this method to replace where ECC is listed as the jurisdiction.
missing_jx <- calls_911 %>%
    group_by(city, jurisdiction) %>%
    summarize(total = n(),
            total_distinct = n_distinct(master_incident_number)) %>%
    slice_max(total, n = 2) %>%
    filter(total > 1)

# Impute missing jurisdictions
calls_911 %<>%
    mutate(incident_jurisdiction = case_when(
                (is.na(jurisdiction) & is.na(city) )~"missing",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="SPPD"]~"SPPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="MAPD"]~"MAPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="RCSO"]~"RCSO",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="SAPD"]~"SAPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="WBPD"]~"WBPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="NBPD"]~"NBPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="NSPD"]~"NSPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="MVPD"]~"MVPD",
                (jurisdiction=="ECC" | is.na(jurisdiction))
                    & city %in% missing_jx$city[missing_jx$jurisdiction=="RVPD" & !is.na(city)]~"RVPD",
                (is.na(jurisdiction)) ~"missing",
                TRUE~jurisdiction)
            )

# Assign policy alignment
calls_911 %<>%
    mutate(policy = case_when(incident_jurisdiction %in% c("SPPD", "RVPD", "MAPD", "SAPD") ~ "Aligned",
                            incident_jurisdiction %in% c("RCSO", "WBPD", "MVPD", "NSPD", "NBPD") ~ "Unchanged Policy",
                            incident_jurisdiction=="FGPD" ~ "Unknown",
                            incident_jurisdiction=="missing" ~ "missing",
                            TRUE ~ "Unknown"),
            agency_approach = case_when(incident_jurisdiction %in% c("MAPD", "NBPD", "SAPD") ~ "Unknown",
                            incident_jurisdiction == "SPPD" ~ "Chief provided guidance",
                            incident_jurisdiction == "RVPD" ~ "Has own policy",
                            incident_jurisdiction %in% c("RCSO", "WBPD", "MVPD", "NSPD") ~ "None",
                            incident_jurisdiction == "missing" ~ "missing",
                            TRUE ~ "Unknown"))

# In the table below, you can compare the jurisdiction and incident_jurisdiction
# rows and where jurisdiction is NA and incident_jurisdiction is not NA you
# can see how many rows were reassigned to that jurisdiction.
calls_911 %>%
    group_by(policy, incident_jurisdiction, jurisdiction) %>%
    summarize(total = n(),
            total_distinct = n_distinct(master_incident_number))

# Most that got reassigned to a jurisdiction were actually cancellations and
# will likely be dropped rows.

# Filtering out calls
# We drop those missing master incident numbers because they are also missing
# a problem description and thus we cannot determine if they concern a crime.
# We also label hang ups and cancellations as these will be dropped during
# some analysis as failed calls. They are included in robustness checks.
calls_911 %<>%
    filter(!is.na(master_incident_number)) %>%
    mutate(is_hangup = ifelse(str_detect(call_disposition, 'Hang-up|Hangup'), TRUE, FALSE),
        is_canceled_call = ifelse(str_detect(call_disposition, 'Cance'), TRUE, FALSE))

write_parquet(calls_911, "path_to_file.parquet")

###########################################################################
# this section of code deals with the traffic stops data.
###########################################################################
df_final_traffic_stops <- read_parquet('path_to_data.parquet')

###########################################################################
# This next section of code deals with parsing information about the stop
# from a text comment field.
# Notes on parsing person information in comments from data provider:
# 1. The Comment contains the Race, Gender, Vehicle Searched, and Person Searched,
# plus the Reason for the stop.
# 2. Race is the first digit of the Code, indicating Race. See the screenshot
# of the mobile CAD clearing screen, which corresponds the race to the number.
# White is 1, Black is 2, Hispanic is 3, Asian is 4, Native American is 5, and
# Other is 6. 6 was not always an option, but was added after 2016.
# 3. Gender is second. M for Male, F for Female or X for Non-binary. X was not
# always an option, but was added after 2019.
# 4. PerSearched appears to be third – Y or N whether the Person was searched.
# 5. VehSearched appears to be fourth – Y or N whether the Vehicle was searched.
# 6. Reason is last – The reason given by the officer for the stop. 1 for Moving
# Violation, 2 for Vehicle Violation, 3 for Investigative, and 4 for 9-1-1/Citizen
# Report. The Reason Code was not always an option, but was added in 2017.
# Example: 1MNY3

# 7. The text is preceded by two colons to make them easier to spot.
# 8. The bracketed number indicated the comment number within the incident.
###########################################################################

# Pre-processing comment text to address common formatting issues with repeated or erroneous delimiters
df_final_traffic_stops %<>%
    mutate(delimiter_dup = str_detect(comment, "[:;,]{3,6}"),
           comment_clean = str_replace_all(comment, "[:;,]{3,6}", "::")) %>%
    mutate(delimiter_alt = str_detect(comment_clean, "[;,]{2}"),
           comment_clean = str_replace_all(comment_clean, "[;,]{2}", "::")) %>%
    mutate(comment_clean = str_to_lower(comment_clean))

# Reviewing changes
df_final_traffic_stops %>% count(delimiter_dup, delimiter_alt) %>% arrange(desc(n))

# There were multiple methods possible for parsing this information from the text.
# Two approaches are provided below that were measured against one another

#############################
# Approach 1: using colon delimiters
# Splitting comment text using delimiters (after cleaning) (note: the regex was developed by Felix Owusu)
df_final_traffic_stops %<>%
    #select(id, master_incident_number, casenumber, problem, comment, comment_clean, vehicle_jurisdiction) %>%
    mutate(comment_split = str_split_fixed(comment_clean, "[;:]{2}", n = 3)) %>%
    mutate(comment_split_1 = comment_split[,1],
           comment_split_2 = comment_split[,2],
           comment_split_3 = comment_split[,3]) %>% # Tested, none generate more than 3 splits
    mutate(comment_split_1 = str_trim(comment_split_1, side = "both"),
           comment_split_2 = str_trim(comment_split_2, side = "both"),
           comment_split_3 = str_trim(comment_split_3, side = "both")) # Trimming leading, trailing white space

# Isolating person/search information in text after delimiter
df_final_traffic_stops %<>%
    mutate(split_2_d1_length = str_locate(comment_split_2, " ")[,1]) %>% # splitting on white space, to keep just the first thing after the delimiter
    mutate(split_2_d1 = if_else(is.na(split_2_d1_length) | split_2_d1_length <= 3, # if there's no space at all in the split or it appears too early to follow demo/search info
                                str_trim(comment_split_2, side = "both"), # Then just reproduce the whole text segment (trimming white space)
                                str_sub(comment_split_2, 1, split_2_d1_length -1))) %>% # otherwise, pull text before space
    mutate(split_2_d1 = str_replace_all(split_2_d1, "[\\].\\',/\\\\]", "")) %>% # Removing extra characters (common typo)
    mutate(delimit_match = split_2_d1 != "") # Marking stops that have demo/search info identified using delimiter parsing method


# Isolating person/search data after 2nd delimiter if present (this is rare)
df_final_traffic_stops %<>%
    mutate(split_3_d1_length = str_locate(comment_split_3, " ")[,1]) %>%
    mutate(split_3_d1 = if_else(is.na(split_3_d1_length) | split_3_d1_length <= 3,
                                str_trim(comment_split_3, side = "both"),
                                str_sub(comment_split_3, 1, split_3_d1_length -1))) %>%
    mutate(split_3_d1 = str_replace_all(split_3_d1, "[\\].\\',/\\\\]", "")) # Removing extra characters (common typo)

#############################
# Approach 2: Using regex
# Regex for matching (note: the regex was developed by Rory Pulvino)
regex_full <- "[\\s:;]\\w{1,2}[yYnN]{2}\\d{0,1}" # With delimiters
regex_parsed <- "\\w{1,2}[yYnN]{2}\\d{0,1}" # Without delimiters

# Identifying matching text in the cleaned comment field
df_final_traffic_stops %<>%
    mutate(demographic_text_regex = str_extract_all(comment_clean, regex_full, simplify = TRUE), # extract all text that matches the data pattern
           regex_match = str_detect(comment_clean, regex_full)) %>% # Marking stops that have demo/search info identified using the regex method
    mutate(regex_d1 = demographic_text_regex[,1],
           regex_d2 = demographic_text_regex[,2]) %>% # saving data separately for instances with multiple persons
    mutate(regex_d1 = str_replace_all(regex_d1, "[:;,]", ""), # removing delimiters
           regex_d2 = str_replace_all(regex_d2, "[:;,]", "")) %>% # removing delimiters
    mutate(regex_d1 = str_trim(regex_d1, side = "both"), # trimming white space
           regex_d2 = str_trim(regex_d2, side = "both"))

# Quickly reviewing results
df_final_traffic_stops %>% count(regex_match)
df_final_traffic_stops %>% count(demographic_text_regex, regex_d1, regex_d2) %>%
arrange(desc(n)) %>%
head(7)
df_final_traffic_stops %>% count(regex_d2) %>% arrange(desc(n)) %>% head(7)

#############################
# comparing coverage of both methods
# Checking coverage by method -- Do the methods identify info in the same stops?
df_final_traffic_stops %>%
    count(delimit_match, regex_match) %>%
    arrange(desc(n))
# Yes, mostly.

# Save length of parsed text from each method for future comparison
df_final_traffic_stops %<>%
    mutate(split_2_d1_length = str_length(split_2_d1),
           regex_d1_length = str_length(regex_d1))

# Reviewing most common codes
df_final_traffic_stops %>%
    count(split_2_d1, regex_d1) %>%
    arrange(desc(n)) %>%
    head(10)

# review sample of extra regex matches
df_final_traffic_stops %>%
    filter(regex_match == TRUE & delimit_match == FALSE) %>%
    select(comment, comment_clean, split_2_d1, demographic_text_regex) %>%
    slice_sample(n = 10)

# review sample of extra delimiter matches
df_final_traffic_stops %>%
    filter(regex_match == FALSE & delimit_match == TRUE) %>%
    select(comment, comment_clean, split_2_d1, demographic_text_regex) %>%
    slice_sample(n = 10)

# reviewing sample of stops with no person information under either method
df_final_traffic_stops %>%
    filter(regex_match == FALSE & delimit_match == FALSE) %>%
    select(comment, comment_clean, split_2_d1, demographic_text_regex) %>% slice_sample(n = 10)

###########################################################################
# Notes regarding findings from different approaches to parsing
# Most stops have info picked up by both methods.
# 1. Regex method picks up info in ~1400 stops where the delimiter method
# does not. Mostly those with no or particularly wonky delimiters
# 2. Delimiter method picks up info in 83 stops where the regex method does
# not. Mostly those with unanticipated typos
# 3. 75 stops have no identifiable person/search information. These appear
# to be the result of typos that alter the field's length/pattern substantially.
# 4. Will need to do some manual cleaning to include these, particularly
# those in the last two categories. Can also just let the regex fly and pick
# up approximate matches. This increases the risk of miscoding, however.
###########################################################################

# additional checks
# Checking text identification by method -- when both methods identify information, does it match?

# Do the parsed values match across methods?
df_final_traffic_stops %<>%
    mutate(method_mismatch_d1 = split_2_d1 != regex_d1 & split_2_d1 != "" & regex_d1 != "")

df_final_traffic_stops %>% count(method_mismatch_d1)
# Almost always match.

# Reviewing mismatches
df_final_traffic_stops %>%
    filter(method_mismatch_d1 == TRUE) %>%
    select(comment, comment_clean, split_2_d1, regex_d1, method_mismatch_d1, split_2_d1_length,
        regex_d1_length) %>%
    slice_sample(n = 20)

# Reviewing mismatches by string length
df_final_traffic_stops %>%
    filter(method_mismatch_d1 == TRUE) %>%
    count(split_2_d1_length, regex_d1_length) %>%
    arrange(desc(n))
# When the methods disagree, its about length. All mismatches have different lengths involved

# When are regex matches longer?
df_final_traffic_stops %>%
    filter(method_mismatch_d1 == TRUE
        & regex_d1_length >= split_2_d1_length) %>%
    select(comment, comment_clean, split_2_d1, regex_d1, method_mismatch_d1,
    split_2_d1_length, regex_d1_length)
# Almost never

# When are delimited matches longer?
df_final_traffic_stops %>%
    filter(method_mismatch_d1 == TRUE
        & regex_d1_length <= split_2_d1_length) %>%
    select(comment, comment_clean, split_2_d1, regex_d1, method_mismatch_d1,
    split_2_d1_length, regex_d1_length) %>%
    slice_sample(n = 20)
# if there's a length mismatch, it's likely in this direction. regex truncates

###########################################################################
# This section reviews the quality of the parsed text based on expected codes
###########################################################################
# Enumerating expected demo/search info values

# Race: 1:6; B, W, A, H, L, N
# Gender: M, F, X
# Person Search: Y, N
# Vehicle search: Y, N
# Reason: 1:4 (sometimes missing in earlier data)

race_cat <- c(1:6)
race_typo <- c("b", "w", "a", "h", "l", "n") # likely typos from people who forget numbers
gender_cat <- c("m", "f", "x")
search_cat <- c("y", "n")
reason_cat <- c("",1:4)

demo_info_list <- expand.grid(race = race_cat,
                            gender = gender_cat,
                            person_search = search_cat,
                            vehicle_search = search_cat,
                            reason = reason_cat) %>%
    mutate(data_entry = paste0(race, gender, person_search, vehicle_search, reason)) %>%
    select(data_entry)


demo_info_list_racetypo <- expand.grid(race = race_typo,
                                    gender = gender_cat,
                                    person_search = search_cat,
                                    vehicle_search = search_cat,
                                    reason = reason_cat) %>%
    mutate(data_entry = paste0(race, gender, person_search, vehicle_search, reason)) %>%
    select(data_entry)


demo_info_list_full <- bind_rows(demo_info_list, demo_info_list_racetypo)

# Note: this could also be a regex, but we appreciate the simplicity and specificity of this approach

# Identifying whether the person info we've identified so far is on the enumerated list of expected entries
df_final_traffic_stops %<>%
    mutate(split_2_d1_match_list = split_2_d1 %in% demo_info_list$data_entry,
           split_3_d1_match_list = split_3_d1 %in% demo_info_list$data_entry,
           regex_d1_match_list = regex_d1 %in% demo_info_list$data_entry,
           regex_d2_match_list = regex_d2 %in% demo_info_list$data_entry) %>%
    mutate(split_2_d1_match_typolist = split_2_d1 %in% demo_info_list_racetypo$data_entry,
           split_3_d1_match_typolist = split_3_d1 %in% demo_info_list_racetypo$data_entry,
           regex_d1_match_typolist = regex_d1 %in% demo_info_list_racetypo$data_entry,
           regex_d2_match_typolist = regex_d2 %in% demo_info_list_racetypo$data_entry)

df_final_traffic_stops %>%
    count(split_2_d1_match_list, regex_d1_match_list) %>% arrange(desc(n))

# Vast majority of the demographic information identified via both methods match expectations
# regex method produces text that matches expectations slightly more often -- manual review
# indicates that this is because it will truncate entries where they are expected to end,
# whereas the delimiter method pulls the full word after the delimiter even if only the first
# portion of the string matches (e.g. the demo info includes trailing extra characters)

# Reviewing mismatches for person 1
split_2_d1_mismatch_df <- df_final_traffic_stops %>%
    filter(split_2_d1 != "" | regex_match == TRUE) %>% # Text pulled from at least 1 method
    filter((split_2_d1_match_list == FALSE & split_2_d1 != "")) %>% # Parsed text doesn't match expectations
    count(comment, comment_clean, split_2_d1, split_2_d1_length, split_2_d1_match_list,
        demographic_text_regex, regex_match) %>%
    arrange(desc(n))

###########################################################################
# Process for creating combined person 1 information using data from both
# parsing methods
###########################################################################
df_final_traffic_stops %<>%
    mutate(
    # First priority: Parsing methods match, info matches expected demo/search format
        match_clean_d1 =
            (delimit_match == TRUE & regex_match == TRUE) & # Both parsing methods work &
            method_mismatch_d1 == FALSE & # Both methods lead to the same text
            (split_2_d1_match_list == TRUE & regex_d1_match_list == TRUE), # Text matches expectations

    # Second priority: Parsing methods match, info matches list with common typos
        match_typo_d1 =
            (delimit_match == TRUE & regex_match == TRUE) & # Both parsing methods work &
            method_mismatch_d1 == FALSE & # Both methods lead to the same text
            (split_2_d1_match_typolist == TRUE & regex_d1_match_typolist == TRUE), # Text matches expectations, with common typos

    # Third priority: Parsing methods match, info does not match expected demo/search format
        match_unexpected_d1 =
            (delimit_match == TRUE & regex_match == TRUE) & # Both parsing methods work &
            method_mismatch_d1 == FALSE & # Both methods lead to the same text
            (split_2_d1_match_list == FALSE & regex_d1_match_list == FALSE &
            split_2_d1_match_typolist == FALSE & regex_d1_match_typolist == FALSE), # Text does not match expectations,

    # Fourth priority: Only one method identifies text
        mismatch_delimit_only_d1 =
            (delimit_match == TRUE & regex_match == FALSE), # Only delimiter method identifies text

        mismatch_regex_only_d1 = (delimit_match == FALSE & regex_match == TRUE), # Only regex method identifies text

    # Fifth priority: Both methods pull, but the methods don't match (we will keep the longer one)
        mismatch_delimit_d1 =
            (delimit_match == TRUE & regex_match == TRUE) & # Both parsing methods work but
            method_mismatch_d1 == TRUE & # Parsing methods lead to different text
            split_2_d1_length > regex_d1_length, # Delimiter method pulls longer string

        mismatch_regex_d1 =
            (delimit_match == TRUE & regex_match == TRUE) & # Both parsing methods work but
            method_mismatch_d1 == TRUE & # Parsing methods lead to different text
            split_2_d1_length < regex_d1_length, # Regex method pulls longer string

    # Sixth priority: Neither method pulls any matching text (just keep full cleaned comment)
        match_no_data_d1 =
            delimit_match == FALSE & regex_match == FALSE
    )

df_final_traffic_stops %>%
    count(match_clean_d1, match_typo_d1, match_unexpected_d1, mismatch_delimit_only_d1,
        mismatch_regex_only_d1, mismatch_delimit_d1, mismatch_regex_d1, match_no_data_d1) %>%
    arrange(desc(n))
# These should be exhaustive and mutually exclusive

# Creating combined person 1 demographic info field
df_final_traffic_stops %<>%
    mutate(
        d1_demo_string = case_when(
            match_clean_d1 == TRUE ~ split_2_d1, # when everything's good, pull from whichever
            match_typo_d1 == TRUE ~ split_2_d1, # when methods match and include anticipated typos, pull from whichever
            match_unexpected_d1 == TRUE ~ split_2_d1, # when methods match but we don't meet expectations, pull from whichever but review
            mismatch_delimit_only_d1 == TRUE ~ split_2_d1, # when only delimit method is populated, pull that
            mismatch_regex_only_d1 == TRUE ~ regex_d1, # when only regex method is populated, pull that
            mismatch_delimit_d1 == TRUE ~ split_2_d1, # when there is a mismatch and delimit produces longer string, pull that
            mismatch_regex_d1 == TRUE ~ regex_d1, # when there is a mismatch and regex produces longer string, pull that
            match_no_data_d1 == TRUE ~ comment_clean, # when neither match at all, pull the full comment
            TRUE ~ "UNACCOUNTED" # There shouldn't be anything left after this, but here just in case
    ))

# Reviewing those that don't match any method -- in need of manual intervention.
df_final_traffic_stops %>%
    filter(match_no_data_d1 == TRUE) %>%
    select(comment, comment_clean, split_2_d1, regex_d1, method_mismatch_d1, split_2_d1_length,
        regex_d1_length) %>%
    slice_sample(n = 20)

###########################################################################
# Reviewing secondary person stopped information
###########################################################################
df_final_traffic_stops %>%
    filter(split_3_d1 != "" | regex_d2 != "") %>%
    select(comment, split_2_d1, regex_d1, split_3_d1, regex_d2)


df_final_traffic_stops %>%
    #filter(split_3_d1 != "") %>%
    count(split_3_d1_match_list, regex_d2_match_list) %>%
    arrange(desc(n))

df_final_traffic_stops %>%
    filter(regex_d2 != "") %>%
    count(regex_d2) %>%
    arrange(desc(n))

# Reviewing secondary persons identified using delimiters that do not match the expected pattern
d2_mismatch_df <- df_final_traffic_stops %>%
    filter(split_3_d1 != "" | regex_d2 != "") %>% # Secondary person included
    filter(split_3_d1_match_list == FALSE | regex_d2_match_list == FALSE) %>% # Parsed text doesn't match expectations
    select(comment, split_2_d1, regex_d1, split_3_d1, regex_d2)

# Regex seems to be better at identifying these in the case of disagreements.
# The delimited 2nd fields not picked up by regex are improper use of delimiter rather than real demo/search data, so we'll ignore those.

# So we'll use the secondary person info pulled by regex, only in cases where it matches expectations
df_final_traffic_stops %<>%
    mutate(
        match_clean_d2 =
            regex_d2 != "" & # Regex pulls secondary person info
            regex_d2_match_list == TRUE, # Secondary person information matches expectations
        match_typo_d2 =
            regex_d2 != "" & # Regex pulls secondary person info
            regex_d2_match_typolist == TRUE, # Secondary person information matches expectations
    ) %>%
    mutate(
        d2_demo_string = ifelse(match_clean_d2 == TRUE, regex_d2, "")
    )

df_final_traffic_stops %>% count(match_clean_d2, match_typo_d2)

###########################################################################
# Coding the variables based on the text string
###########################################################################
# Coding up demographic and search variables from the strings
df_final_traffic_stops %<>%
    mutate(
        reason_for_stop_1 = case_when(
            str_detect(d1_demo_string, "[yn]{2}1") ~ "Moving Violation",
            str_detect(d1_demo_string, "[yn]{2}2") ~ "Vehicle Violation",
            str_detect(d1_demo_string, "[yn]{2}3") ~ "Investigative",
            str_detect(d1_demo_string, "[yn]{2}4") ~ "Citizen Report",
            TRUE ~ "No Reason Given"),

        race_1 = case_when(
            str_detect(d1_demo_string, "[1][mfx][yn]{2}") ~ "White",
            str_detect(d1_demo_string, "[2][mfx][yn]{2}") ~ "Black",
            str_detect(d1_demo_string, "[3][mfx][yn]{2}") ~ "Hispanic",
            str_detect(d1_demo_string, "[4][mfx][yn]{2}") ~ "Asian",
            str_detect(d1_demo_string, "[5][mfx][yn]{2}") ~ "Native Am",
            str_detect(d1_demo_string, "[6][mfx][yn]{2}") ~ "Other",
            TRUE ~ "Unknown"),

        race_code_1 = case_when(
            str_detect(d1_demo_string, "[123456][mfx][yn]{2}") ~ "Numerically Coded",
            str_detect(d1_demo_string, "[w][mfx][yn]{2}") ~ "White",
            str_detect(d1_demo_string, "[b][mfx][yn]{2}") ~ "Black",
            str_detect(d1_demo_string, "[hl][mfx][yn]{2}") ~ "Hispanic",
            str_detect(d1_demo_string, "[a][mfx][yn]{2}") ~ "Asian or African American",
            str_detect(d1_demo_string, "[n][mfx][yn]{2}") ~ "Native Am",
            str_detect(d1_demo_string, "[o][mfx][yn]{2}") ~ "Other",
            TRUE ~ "Unknown"),

        gender_1 = case_when(
            str_detect(d1_demo_string, "[123456][m][yn]{2}") ~ "Male",
            str_detect(d1_demo_string, "[123456][f][yn]{2}") ~ "Female",
            str_detect(d1_demo_string, "[123456][x][yn]{2}") ~ "Non-binary",
            TRUE ~ "Unknown"),

        person_searched_1 = str_detect(d1_demo_string, "[123456][mfx][y][yn]"),
        vehicle_searched_1 = str_detect(d1_demo_string, "[123456][mfx][yn][y]")
    ) %>%
    # For secondary persons
    mutate(
        reason_for_stop_2 = case_when(
            str_detect(d2_demo_string, "[yn]{2}1") ~ "Moving Violation",
            str_detect(d2_demo_string, "[yn]{2}2") ~ "Vehicle Violation",
            str_detect(d2_demo_string, "[yn]{2}3") ~ "Investigative",
            str_detect(d2_demo_string, "[yn]{2}4") ~ "Citizen Report",
            TRUE~"No Reason Given"),

        race_2 = case_when(
            str_detect(d2_demo_string, "[1][mfx][yn]{2}") ~ "White",
            str_detect(d2_demo_string, "[2][mfx][yn]{2}") ~ "Black",
            str_detect(d2_demo_string, "[3][mfx][yn]{2}") ~ "Hispanic",
            str_detect(d2_demo_string, "[4][mfx][yn]{2}") ~ "Asian",
            str_detect(d2_demo_string, "[5][mfx][yn]{2}") ~ "Native Am",
            str_detect(d2_demo_string, "[6][mfx][yn]{2}") ~ "Other",
            TRUE ~ "Unknown"),

        race_code_2 = case_when(
            str_detect(d2_demo_string, "[123456][mfx][yn]{2}") ~ "Numerically Coded",
            str_detect(d2_demo_string, "[w][mfx][yn]{2}") ~ "White",
            str_detect(d2_demo_string, "[b][mfx][yn]{2}") ~ "Black",
            str_detect(d2_demo_string, "[hl][mfx][yn]{2}") ~ "Hispanic",
            str_detect(d2_demo_string, "[a][mfx][yn]{2}") ~ "Asian or African American",
            str_detect(d2_demo_string, "[n][mfx][yn]{2}") ~ "Native Am",
            str_detect(d2_demo_string, "[o][mfx][yn]{2}") ~ "Other",
            TRUE ~ "Unknown"),

        gender_2 = case_when(
            str_detect(d2_demo_string, "[123456][m][yn]{2}") ~ "Male",
            str_detect(d2_demo_string, "[123456][f][yn]{2}") ~ "Female",
            str_detect(d2_demo_string, "[123456][x][yn]{2}") ~ "Non-binary",
            TRUE ~ "Unknown"),

        person_searched_2 = str_detect(d2_demo_string, "[123456][mfx][y][yn]"),
        vehicle_searched_2 = str_detect(d2_demo_string, "[123456][mfx][yn][y]")
    )

    # Reviewing results
df_final_traffic_stops %>%
    count(reason_for_stop_1) %>%
    arrange(desc(n))
df_final_traffic_stops %>%
    count(race_1) %>%
    arrange(desc(n))
df_final_traffic_stops %>%
    count(race_code_1) %>%
    arrange(desc(n))
df_final_traffic_stops %>%
    count(gender_1) %>%
    arrange(desc(n))
df_final_traffic_stops %>%
    count(person_searched_1, vehicle_searched_1) %>%
    arrange(desc(n))

# Note: Felix checked these numbers against Rory Pulvino's initial approach -- overall numbers look
# quite similar, with some differences across a few hundred observations due to addl. data cleaning
# and slightly more restrictive regex in current approach.

###########################################################################
# Assigning missing police departments and police department alignment
###########################################################################
# Where the incident_jurisdiction is listed as ECC, this is early data entry error.
# ECC should be replaced with the agency assigned for that city.

# determine what the most common agency is per city listed as ECC
ECC_cities <- df_final_traffic_stops$city[df_final_traffic_stops$vehicle_jurisdiction=='ECC']
df_final_traffic_stops %>%
    filter(city %in% ECC_cities) %>%
    group_by(city, vehicle_jurisdiction) %>%
    summarize(Total = n()) %>%
    mutate(top = ifelse(Total==max(Total), 1, 0)) %>%
    filter(top==1)

df_final_traffic_stops %<>%
    mutate(vehicle_jurisdiction = case_when(vehicle_jurisdiction=="ECC"
                                            & city %in% c("St Paul", "ST. PAUL", "Fort Snelling",
                                                            "Minneapolis", "South St Paul")~"SPPD",
                                            vehicle_jurisdiction=="ECC"
                                            & city %in% c("Maplewood")~"MAPD",
                                            vehicle_jurisdiction=="ECC"
                                            & city %in% c("Newport")~"RCSO",
                                            vehicle_jurisdiction=="ECC"
                                            & city %in% c("St Anthony", "lauderdale")~"SAPD",
                                            TRUE~vehicle_jurisdiction)
    )

# Assigning policy alignment based on department
df_final_traffic_stops %<>%
    mutate(policy = case_when(vehicle_jurisdiction %in% c("SPPD", "RVPD", "MAPD", "SAPD") ~ "Aligned",
                            vehicle_jurisdiction %in% c("RCSO", "WBPD", "MVPD", "NSPD", "NBPD") ~ "Unchanged Policy",
                            vehicle_jurisdiction=="FGPD" ~ "Unknown",
                            TRUE ~ "Unknown"),
            agency_approach = case_when(vehicle_jurisdiction %in% c("MAPD", "NBPD", "SAPD") ~ "Unknown",
                            vehicle_jurisdiction == "SPPD" ~ "Chief provided guidance",
                            vehicle_jurisdiction == "RVPD" ~ "Has own policy",
                            vehicle_jurisdiction %in% c("RCSO", "WBPD", "MVPD", "NSPD") ~ "None",
                            TRUE ~ "Unknown"))

write_parquet(df_final_traffic_stops, "path_to_file.parquet")

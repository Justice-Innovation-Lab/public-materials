The report this code pertains to can be found [here](https://traffic-stop-policy-ramsey-county.justiceinnovationlab.org/).

## Code Explanation
Analysis for this report spanned several months and included mulitple data sources and interim reports that were provided to the jurisdiction. We partnered with Felix Owusu during our initial exploration of the data and initially determined to use R for the project. The beginning stages of exploring and cleaning the data then were mostly done in R. That code was compiled into a single file `clean_data.r` included here. The later analysis and production of visuals that were used in the public report was done in python. That code was compiled into a single file `report_analysis.py`. The remaining file, `clean_data.py` is compiled code of our geoprocessing of the data that we did in python from the start given that it was computationally faster to do so. 

The main source of data for this project was the Ramsey County Emergency Communications Center (ECC). Other data sources included open source data for crime in 
St. Paul and data provided by the St. Paul Police Department.

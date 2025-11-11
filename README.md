# Traffic Risk Mapping

## Data Acquisition
- **Source**: Motor Vehicle Collisions from the City of New York (https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
- **Description**: Details on crash events that occured in New York City. Each row represents a crash event. (2.22M rows)

## Data Preparation & Cleaning
- Loaded Dataset: NewYork_collisions_raw_20251110.csv
- Canonicalize Columns
- Drop rows where street name or contributing factor 1 are NaN or none
- Remove unessecary columns
- Handle errors with coordinate information (Numerice to Invalids, exact zeros, outside bounds)
- Combine date and time plus feature engineering
- Conversion to geodataframe
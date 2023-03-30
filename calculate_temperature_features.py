import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import xarray as xr

def imp_db (month, year):
	if month < 10:
		nc = xr.open_dataset("../raw_data/raw_temps/tas_Athens_UrbClim_"+str(year)+"_0"+str(month)+"_v1.0.nc").to_dataframe().reset_index()
	else:
		nc = xr.open_dataset("../raw_data/raw_temps/tas_Athens_UrbClim_"+str(year)+"_"+str(month)+"_v1.0.nc").to_dataframe().reset_index()
	nc = nc.drop(columns = ['x','y'])
	nc['day'] = pd.to_datetime(nc['time']).dt.day
	nc['year'] = pd.to_datetime(nc['time']).dt.year
	nc['month'] = pd.to_datetime(nc['time']).dt.month
	nc['time'] = pd.to_datetime(nc['time']).dt.hour
	return(nc)

def get_temp_features(gdf2,hot_day_cutoff):
    bounds = gdf2.to_crs(epsg=4326).total_bounds
    gdf_list = []
    for i in range(6,9):
        nc = imp_db(i,2017) #Change this if you're not using nc files from 2017
        nc['tas'] = nc['tas'] - 273.15
        filtered_nc = nc[(nc['latitude'] < bounds[3]+0.01) & (nc['latitude'] > bounds[1]-0.01) & (nc['longitude'] < bounds[2]+0.01) & (nc['longitude'] > bounds[0]-0.01)]
        gdf = gdf2.copy()
        df = filtered_nc.copy()
        # Assuming your dataframe is called 'df' and geodataframe is called 'gdf'

        # Convert the dataframe to a geodataframe
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        df_geo = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:4326')
        df_geo = df_geo.to_crs(epsg=2100)

        # Perform a spatial join between the dataframes
        df_joined = gpd.sjoin(df_geo, gdf, op='within')

        # Calculate the required values for each zipcode
        grouped = df_joined.groupby('MOXAP')
        max_temperature = grouped['tas'].max()
        mean_temperature = grouped['tas'].mean()

        # Group by day and find the maximum temperature for each day
        df_joined['date'] = pd.to_datetime(df_joined[['year', 'month', 'day']])
        grouped_daily = df_joined.groupby(['MOXAP', 'date'])

        # Calculate hot days by checking if the maximum temperature for each day is above 38 degrees
        hot_days = grouped_daily['tas'].max().reset_index()
        hot_days['is_hot_day'] = hot_days['tas'] > hot_day_cutoff
        hot_days_count = hot_days.groupby('MOXAP')['is_hot_day'].sum()

        # Create a new dataframe with the calculated values
        zipcode_stats = pd.DataFrame({
            'MOXAP': max_temperature.index,
            'max_temperature': max_temperature.values,
            'mean_temperature': mean_temperature.values,
            'hot_days': hot_days_count.values
        })

        # Merge the results back to the geodataframe
        gdf_final = gdf.merge(zipcode_stats, on='MOXAP')
        gdf_list.append(gdf_final)
    #Now we have a list of 3 geodataframes, one for each month, we want to sum 'hot_days', max 'max_temperature' and mean 'mean_temperature'
    gdf_1 = gdf_list[0].copy()
    gdf_2 = gdf_list[1].copy()
    gdf_3 = gdf_list[2].copy()
    mean_temp = (gdf_1['mean_temperature'] + gdf_2['mean_temperature'] + gdf_3['mean_temperature'])/3
    max_temp = pd.DataFrame([gdf_1['max_temperature'],gdf_2['max_temperature'],gdf_3['max_temperature']]).max(axis=0)
    hot_days_sum = gdf_1['hot_days'] + gdf_2['hot_days'] + gdf_3['hot_days']
    gdf_1['mean_temperature'] = mean_temp
    gdf_1['max_temperature'] = max_temp
    gdf_1['hot_days'] = hot_days_sum
    return gdf_1

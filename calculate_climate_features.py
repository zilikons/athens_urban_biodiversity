import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import xarray as xr
import numpy as np

def imp_db (month, year, feature='tas'):
	if month < 10:
		nc = xr.open_dataset(f"../raw_data/raw_temps/{feature}_Athens_UrbClim_"+str(year)+"_0"+str(month)+"_v1.0.nc").to_dataframe().reset_index()
	else:
		nc = xr.open_dataset(f"../raw_data/raw_temps/{feature}_Athens_UrbClim_"+str(year)+"_"+str(month)+"_v1.0.nc").to_dataframe().reset_index()
	nc = nc.drop(columns = ['x','y'])
	nc['day'] = pd.to_datetime(nc['time']).dt.day
	nc['year'] = pd.to_datetime(nc['time']).dt.year
	nc['month'] = pd.to_datetime(nc['time']).dt.month
	nc['time'] = pd.to_datetime(nc['time']).dt.hour
	return(nc)

def get_temp_features(gdf2,hot_day_cutoff=38):
    bounds = gdf2.to_crs(epsg=4326).total_bounds
    gdf_list = []
    for i in range(6,9):
        nc = imp_db(i,2017,'tas') #Change this if you're not using nc files from 2017
        nc['tas'] = nc['tas'] - 273.15
        filtered_nc = nc[(nc['latitude'] < bounds[3]+0.01) & (nc['latitude'] > bounds[1]-0.01) & (nc['longitude'] < bounds[2]+0.01) & (nc['longitude'] > bounds[0]-0.01)]
        gdf = gdf2.copy()
        df = filtered_nc.copy()
        # Assuming your dataframe is called 'df' and geodataframe is called 'gdf'

        # Convert the dataframe to a geodataframe
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        df_geo = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:4326')
        df_geo = df_geo.to_crs(gdf.crs)

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

def get_humidity_features(gdf2):
    bounds = gdf2.to_crs(epsg=4326).total_bounds
    gdf_list = []
    for i in range(6,9):
        nc = imp_db(i,2017,'huss') #Change this if you're not using nc files from 2017
        filtered_nc = nc[(nc['latitude'] < bounds[3]+0.01) & (nc['latitude'] > bounds[1]-0.01) & (nc['longitude'] < bounds[2]+0.01) & (nc['longitude'] > bounds[0]-0.01)]
        gdf = gdf2.copy()
        df = filtered_nc.copy()
        # Assuming your dataframe is called 'df' and geodataframe is called 'gdf'

        # Convert the dataframe to a geodataframe
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        df_geo = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:4326')
        df_geo = df_geo.to_crs(gdf.crs)

        # Perform a spatial join between the dataframes
        df_joined = gpd.sjoin(df_geo, gdf, op='within')

        # Calculate the required values for each zipcode
        grouped = df_joined.groupby('MOXAP')
        max_humidity = grouped['huss'].max()
        mean_humidity = grouped['huss'].mean()

        # Group by day and find the maximum temperature for each day
        # df_joined['date'] = pd.to_datetime(df_joined[['year', 'month', 'day']])
        # grouped_daily = df_joined.groupby(['MOXAP', 'date'])

        # Calculate hot days by checking if the maximum temperature for each day is above 38 degrees
        # hot_days = grouped_daily['tas'].max().reset_index()
        # hot_days['is_hot_day'] = hot_days['tas'] > hot_day_cutoff
        # hot_days_count = hot_days.groupby('MOXAP')['is_hot_day'].sum()

        # Create a new dataframe with the calculated values
        zipcode_stats = pd.DataFrame({
            'MOXAP': max_humidity.index,
            'max_humidity': max_humidity.values,
            'mean_humidity': mean_humidity.values,
        })

        # Merge the results back to the geodataframe
        gdf_final = gdf.merge(zipcode_stats, on='MOXAP')
        gdf_list.append(gdf_final)
    #Now we have a list of 3 geodataframes, one for each month, we want to sum 'hot_days', max 'max_temperature' and mean 'mean_temperature'
    gdf_1 = gdf_list[0].copy()
    gdf_2 = gdf_list[1].copy()
    gdf_3 = gdf_list[2].copy()
    mean_humi = (gdf_1['mean_humidity'] + gdf_2['mean_humidity'] + gdf_3['mean_humidity'])/3
    max_humi = pd.DataFrame([gdf_1['max_humidity'],gdf_2['max_humidity'],gdf_3['max_humidity']]).max(axis=0)
    # hot_days_sum = gdf_1['hot_days'] + gdf_2['hot_days'] + gdf_3['hot_days']
    gdf_1['mean_humidity'] = mean_humi
    gdf_1['max_humidity'] = max_humi
    # gdf_1['hot_days'] = hot_days_sum
    return gdf_1


def q_sat(T, p):
    Tc = T - 273.15     # temperature in Celsius
    e_sat = 611.2 * np.exp(17.67 * Tc / (Tc + 243.15))
    return e_sat * 0.622 / (p - e_sat)
def wbt_isobaric(T, h, p, h_type='s', p_type='sf'):

    # T: [dry bulb] temperature (K)
    # h: humidity (specific, relative, or dewpoint)
    #    types: relative ['r'], specific ['s'], or dewpoint ['d'])
    # p: pressure in Pa
    #    types: surface ['sf'] or sea level ['sl']
    # q: specific humidity (mass mixing ratio)
    # ps: surface pressure in Pa
    # Z: global constant for surface height
    #
    cp = 1005.7         # specific heat of dry air
    L0 = 2.501e6        # latent heat of vaporization (at 273.15K)
    l = 0.00237e6       # temperature dependence of latent heat
    g = 9.80616
    Ra = 287.
    gamma = -0.0065
    #
    if p_type == 'sf':
        ps = p
    else:
        pass
        # convert sea level pressure to surface pressure
        # (when surface pressure is not available)
        #ps = p * (1 - gamma*Z/T)**(g/Ra/gamma)
    # Note that due to exponential shape of Clausius-Clayperon relation
    # and associated strong non-linearity,
    # relative humidity is not appropriate for daily-averaged fields,
    # only valid for instantaneous fields
    if h_type == 'r':
        q0 = q_sat(T, ps) * h           # relative humidity
        ind_sat = (h >= 1.0)            # index for saturated points
    elif h_type == 's':
        q0 = h                          # specific humidity
        ind_sat = (h >= q_sat(T, ps))   # index for saturated points
    elif h_type == 'd':
        q0 = q_sat(h, ps)				# dewpoint temperature
        ind_sat = (h >= T)				# index for saturated points
    else:
        print('Please provide a valid flag for humidity (r-relative, s-specific, d-dewpoint T)')
    # bisection method
    T1 = T - L0 * (q_sat(T, ps) - q0) / cp
    T2 = T.copy()       # must use copy or T will change
    n = 0
    while np.max(T2 - T1) > 1e-4:
        Tm = (T1 + T2) / 2
        q = q_sat(Tm, ps)        # saturated specific humidity at Tm
        ind1 = (cp * (T - Tm) >= L0 * (q - q0))
        ind2 = ~ind1
        T1[ind1] = Tm[ind1]
        T2[ind2] = Tm[ind2]
        n += 1
    # print(n)
    Tw = Tm
    Tw[ind_sat] = T[ind_sat]
    return Tm

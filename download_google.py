import os
import googlemaps
import random
import requests
from datetime import datetime

api_key = 'AIzaSyDc8eBVhaouyt_v3ml_cDUwMvnZiY9R4t0'


gm = googlemaps.Client(key=api_key)


geocode_result=gm.geocode('new york city')[0]
print geocode_result

geocode_result['geometry']

geocode_result['geometry']['location'] #will get the lattitude and longitude
lat_north=geocode_result['geometry']['bounds']['northeast']['lat']
lat_south=geocode_result['geometry']['bounds']['southwest']['lat']

long_west=geocode_result['geometry']['bounds']['southwest']['lng']
long_east=geocode_result['geometry']['bounds']['northeast']['lng']


url_base ='https://maps.googleapis.com/maps/api/staticmap?center='
url_street='&zoom=15&size=256x256&style=element:labels|visibility:off&key='+api_key
url_satellite='&zoom=15&size=256x256&maptype=satellite&key='+api_key





for i in range(1, 1025):
    center = str(random.uniform(lat_south, lat_north))+','+ str(random.uniform(long_west, long_east))
    img = open('google_data/train/map_std/std_'+str(i)+'.png','wb')
    img.write(requests.get(url_base+'center='+center+url_street).content)
    img.close()

    img = open('google_data/train/sat/sat_'+str(i)+'.png','wb')
    img.write(requests.get(url_base+'center='+center+url_satellite).content)
    img.close()

#!/usr/bin/env python

import os
import sys
import numpy as np
import shapefile
from datetime import datetime, timedelta
from opendrift.readers import reader_ROMS_native_MOANA
from opendrift.models.fishlarvaeorient import FishLarvaeOrient
import time
start_time = time.time()

####
# Month selection
####

runtime = [datetime(2016,3,2), datetime(2016,3,4)]
finish_time = (datetime(2016,3,2) + timedelta(days = 31.0))


###############################
# MODEL SELECTION
###############################
o = FishLarvaeOrient(loglevel=50)
o.max_speed = 3#
###############################
# READERS
###############################
thredds_path_1 = 'http://thredds.moanaproject.org:8080/thredds/dodsC/moana/ocean/NZB/v1.9/monthly_avg/nz5km_avg_201603.nc'
reader_moana_v19 = reader_ROMS_native_MOANA.Reader(thredds_path_1) # load data for that year
reader_moana_v19.multiprocessing_fail = True # bypass the use of multi core for coordinates conversion and seems to make the model run much faster.

# use native landmask of ROMS files
o.add_reader([reader_moana_v19]) # [reader_landmask,reader_moana_v19]
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask

###############################
# PARTICLE SEEDING
###############################
# generic point release location for test

# Define the starting position of the particles
nb_parts = 1
points = np.loadtxt('./Release_centroid_nat_dist_paua.xyz', delimiter='\t', dtype=str) # Text file with lon and lat of release points
plon = points[:,0].astype(np.float)
plat = points[:,1].astype(np.float)
tot_parts = nb_parts * len(plon) # total number of particles released
plon = np.tile(plon, nb_parts)
plat = np.tile(plat, nb_parts)

# Define the release time. author: Calvin Quigley (19/04/2021)
def create_seed_times(start, end, delta):
  """
  crate times at given interval to seed particles
  """
  out = []
  start_t = start
  end_t = datetime.strptime(str(end), "%Y-%m-%d %H:%M:%S")
  while start_t < end:
    out.append(start_t) 
    start_t += delta
  return out

times = create_seed_times(runtime[0], 
                          runtime[1], timedelta(days = 1))

# Define release depth
z = np.random.uniform(-0.5,-8,size=tot_parts) # generate random depth
#z = 'seafloor+1'

# Seed particles
for i in range(len(times)):
	o.seed_elements(plon, plat, number=tot_parts, z=z, time = times[i], terminal_velocity=0.0)

# seed
o.set_config('seed:ocean_only', True) # keep only particles from the "frame" that are on the ocean

###############################
# PHYSICS
###############################
o.set_config('environment:fallback:x_wind', 0.0)
o.set_config('environment:fallback:y_wind', 0.0)
o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
o.set_config('environment:fallback:y_sea_water_velocity', 0.0)
o.set_config('environment:fallback:sea_floor_depth_below_sea_level', 12000)

# horizontal and vertical diffusion
Kxy = 0.1176 # m2/s-1
Kz = 0.01 # m2/s-1
o.set_config('environment:fallback:ocean_vertical_diffusivity', Kz) # specify constant ocean_vertical_diffusivity in m2.s-1
o.set_config('vertical_mixing:diffusivitymodel', 'constant') # constant >> use fall back values o.set_config('environment:fallback:ocean_vertical_diffusivity'] = Kz for all profile
# can be environment (i.e. from reader), or  windspeed_Large1994 ,windspeed_Sundby1983, 
o.set_config('drift:horizontal_diffusivity',Kxy) # using new config rather than current uncertainty
# seed
o.set_config('seed:ocean_only', True) # keep only particles from the "frame" that are on the ocean
# drift
o.set_config('drift:advection_scheme','runge-kutta4') # or 'runge-kutta'
o.set_config('drift:current_uncertainty', 0.0 ) # note current_uncertainty can be used to replicate an horizontal diffusion spd_uncertain = sqrt(Kxy*2/dt)  
o.set_config('drift:vertical_advection', True)
o.set_config('vertical_mixing:timestep', 90.0)  # if some ocean_vertical_diffusivity!=0, turbulentmixing:timestep should be << 900 seconds
o.set_config('drift:vertical_mixing', True) 

###############################
# Type of settlement
###############################
o.habitat('./habitat/National_distribution_paua.shp') # Location of the shapefile with the habitat
o.set_config('biology:settlement_in_habitat', True)
o.set_config('drift:max_age_seconds', 30*24*3600) # 
o.set_config('biology:min_settlement_age_seconds', 22*24*3600) # Beginning of the competency period
o.set_config('general:seafloor_action', 'lift_to_seafloor')

###############################
# Vertical swimming
###############################
# vertical swimming speed and larval development stages
# enable the OVM for the larvae. Need to turn on the vertical mixing (either constant or environment)
o.set_config('biology:OVM', True)
o.set_config('biology:vertical_migration_speed_constant', 0.001) # Vertical swimming speed in meters/s. Used in Ontogenetic Vertical Migration
o.set_config('biology:pre_flexion', 1.0*24*3600)# Beginning of the pre-flexion stage, 24hr after hatching
o.set_config('biology:flexion', 15.0*24*3600)# Beginning of the flexion stage. Beginning of the orientation
o.set_config('biology:post_flexion', 18.0*24*3600)# Beginning of the post-flexion stage
# Preferred depth per development stage
o.set_config('biology:depth_early_stage', -10.0)# Preferred depth in meters
o.set_config('biology:depth_pre_flexion', -15.0)
o.set_config('biology:depth_flexion', -20.0)
o.set_config('biology:depth_post_flexion', -30.0)
o.set_config('drift:maximum_depth', -100.0) # maximum depth of dispersal in meters (negative)

###############################
# Orientation
###############################
# Larval orientation toward the nearest habitat
o.habitat('./habitat/National_distribution_paua.shp') # Location of the shapefile with the habitat
o.set_config('biology:direct_orientation', True) # orient toward the nearest reef
o.set_config('biology:flexion', 15.0*24*3600)# Beginning of the flexion stage. Beginning of the orientation => if using OVM, values must agree
o.set_config('biology:max_orient_distance', 10.0)# Orientation distance in kilometers. Maximum distance at which the habitat is detected
# Horizontal swimming speed:
o.set_config('biology:hatch_swimming_speed', 2.0)# Swimming speed at hatching, in cm/s
o.set_config('biology:settle_swimming_speed', 30.0)# Swimming speed at settlement, in cm/s

o.list_config()
# o.list_configspec()

###############################
# RUN 
###############################
# Running model (until end of driver data)
o.run(stop_on_error=False,
      time_step=timedelta(seconds = 900), # requires a small time-step to compute the orientation
      end_time = finish_time,
      time_step_output=timedelta(minutes = 1200),
      outfile= 'test_development_orientation_Opendrift_using_moana.nc')


print(o)
o.plot(fast=True, color='z', corners=[163, 180, -52, -31])
o.animation(fast=True, color='z', corners=[163, 180, -52, -31])
o.animation_profile()

#!/usr/bin/env python

import os
import sys
import numpy as np
import shapefile
from datetime import datetime, timedelta
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.fishlarvaeorient import FishLarvaeOrient
import time
start_time = time.time()

####
# Month selection
####

runtime = [datetime(2019,6,1), datetime(2019,6,3)]
finish_time = (datetime(2019,6,3) + timedelta(days = 35.0))


###############################
# MODEL SELECTION
###############################
o = FishLarvaeOrient(loglevel=50)
o.max_speed = 3#

###############################
# READERS
###############################
thredds_path_1 = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
reader = reader_netCDF_CF_generic.Reader(thredds_path_1) # load data for that year
#reader.multiprocessing_fail = True
o.add_reader([reader]) # [reader_landmask,reader_moana_v19]
# No native landmask on HYCOM
o.set_config('general:use_auto_landmask', True) # prevent opendrift from making a new dynamical landmask

###############################
# PARTICLE SEEDING
###############################
# Define the starting position of the particles in the FLK
nb_parts = 20
points = np.loadtxt('./release_FLK.xyz', delimiter='\t', dtype=str)
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
	o.seed_elements(plon, plat, number=tot_parts, z=z, time = times[i])

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
o.habitat('./habitat/Polygons_reefs.shp') # Location of the shapefile with the habitat
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
o.set_config('biology:vertical_migration_speed_constant', 0.01) # Vertical swimming speed in meters/s. Used in Ontogenetic Vertical Migration
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
o.habitat('./habitat/Polygons_reefs.shp') # Location of the shapefile with the habitat
o.set_config('biology:orientation', 'direct') # orientation of the larvae: direct, rheotaxis, cardinal, continuous_1, continuous_2, or none
o.set_config('biology:beginning_orientation', 15.0*24*3600)# Beginning of the flexion stage. Beginning of the orientation => if using OVM, values must agree
o.set_config('biology:max_orient_distance', 10.0)# Orientation distance in kilometers. Maximum distance at which the habitat is detected
o.set_config('biology:cardinal_heading', 180.0)# Cardinal heading for larval orientation when 'cardinal' option is selected
# Horizontal swimming speed:
o.set_config('biology:hatch_swimming_speed', 2.0)# Swimming speed at hatching, in cm/s
o.set_config('biology:settle_swimming_speed', 30.0)# Swimming speed at settlement, in cm/s

o.list_config()

###############################
# RUN 
###############################
# Running model (until end of driver data)
o.run(stop_on_error=False,
      time_step=timedelta(seconds = 900), # requires a small time-step to compute the orientation
      end_time = finish_time,
      time_step_output=timedelta(seconds = 3600 * 1),
      outfile= 'test_development_orientation_Opendrift_FLK.nc')


print(o)
o.plot(fast=True, corners=[-84.5, -79, 23, 27.5])
o.animation(fast=True, color='status', corners=[-84.5, -79, 23, 27.5])
o.animation_profile()

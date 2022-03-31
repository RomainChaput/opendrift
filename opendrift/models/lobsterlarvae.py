# This file is intended for Jasus edwardsii and has been adapted from bivalvelarvae.py and larvalfish.py 
# It introduces a multi-step dispersal phase based on the development of the lobster larvae: nauplosioma, phyllosoma, and puerulus
# . Nauplosioma stage: lobster larvae hatch into this stage and rise to the surface where they metamorphose into phyllosoma. Brief stage (<12hrs) that is not represented in this code, but can be implied by the release of larvae slightly offshore and at the surface  
# . Phyllosoma stage: planktonic stage where larvae can control their vertical position in the water column via buoyancy. Late stage is characterized by diel vertical migration and slow swimming. Rarely found inshore after the stage 5, so we remove phyllosoma larvae found within 20 km of the coast 4 to 12 mo after hatching.
# . Puerulus stage: at this stage the lobster larvae have developed horizontal swimming capabilities and if close enough to the coast will swim to their settlement habitat. The pueruli have poorly developed mouth and rely on stored energy for the final stretch of their dispersal.
# 
#  Authors : Romain Chaput, University of Wellington, Simon Weppe MetOcean Solutions/MetService New Zealand
# 
# Under development - more testing to do.
# Stable version

import numpy as np
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray
import logging; logger = logging.getLogger(__name__)
from datetime import timezone
import fiona
from shapely.geometry import Polygon, Point, MultiPolygon,asPolygon # added for settlement in polygon only
import shapely
import random
from sklearn.neighbors import BallTree

# Defining the  element properties from Pelagicegg model
class LobsterLarvaeObj(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for pelagic eggs/larvae
    """

    variables = Lagrangian3DArray.add_variables([
                ('neutral_buoyancy_salinity', {'dtype': np.float32,
                                       'units': '[]',
                                       'default': 31.25}),
                ('age_seconds', {'dtype': np.float32,
                         'units': 's',
                         'default': 0.}),
                ('density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),
                ('terminal_velocity', {'dtype': np.float32,
                       'units': 'm/s',
                       'default': 0.}),
                ('light', {'dtype': np.float32,
                             'units': 'ugEm2',
                             'default': 0.}),
                ('metamorphosis', {'dtype': np.float32,
                             'units': '[]',
                             'default': 0.}),
                ('u_swim', {'dtype': np.float32,
                             'units': 'm/s',
                             'default': 0.}),
                ('v_swim', {'dtype': np.float32,
                             'units': 'm/s',
                             'default': 0.})])

class LobsterLarvae(OceanDrift):
    """
    LobsterLarvae inherits from the bivalve larvae module which includes a minimum age before 
    settling on coast or seafloor can occur, as well as the option to define suitable habitat polygons
    """

    ElementType = LobsterLarvaeObj

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_wave_significant_height': {'fallback': 0},
        'sea_ice_area_fraction': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 100},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'sea_water_temperature': {'fallback': 15, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True},
        'sea_surface_height': {'fallback': 0.0},
        'surface_downward_x_stress': {'fallback': 0},
        'surface_downward_y_stress': {'fallback': 0},
        'turbulent_kinetic_energy': {'fallback': 0},
        'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'ocean_mixed_layer_thickness': {'fallback': 50},
      }

    # Vertical profiles of the following parameters will be available in
    # dictionary self.environment.vertical_profiles
    # E.g. self.environment_profiles['x_sea_water_velocity']
    # will be an array of size [vertical_levels, num_elements]
    # The vertical levels are available as
    # self.environment_profiles['z'] or
    # self.environment_profiles['sigma'] (not yet implemented)
    required_profiles = ['sea_water_temperature',
                         'sea_water_salinity',
                         'ocean_vertical_diffusivity']
    required_profiles_z_range = [-200, 0] # The depth range (in m) which
                                          # profiles shall cover

    # Default colors for plotting
    status_colors = {'initial': 'green', 'active': 'blue',
                     'settled_on_coast': 'red', 'died': 'yellow', 
                     'settled_on_bottom': 'magenta',
                     'settled_on_habitat': 'orange',
                     'swam_too_close_to_shore' : 'black'}

    def __init__(self, *args, **kwargs):
        
        # Calling general constructor of parent class
        super(LobsterLarvae, self).__init__(*args, **kwargs)

        # By default, larvae do not strand when reaching shoreline
        # They are recirculated back to previous position instead
        self.set_config('general:coastline_action', 'previous') # same as bivalve
        # resuspend larvae that reach seabed by default 
        self.set_config('general:seafloor_action', 'lift_to_seafloor') # same as bivalve

        ##add config spec
        # habitat
        self._add_config({ 'biology:min_settlement_age_seconds': {'type': 'float', 'default': 0.0,'min': 0.0, 'max': 1.0e10, 'units': 'seconds',
                           'description': 'minimum age in seconds at which larvae can start to settle on seabed or stick to shoreline',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:settlement_in_habitat': {'type': 'bool', 'default': False,
                           'description': 'settlement restricted to suitable habitat only',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:direct_orientation_habitat': {'type': 'bool', 'default': False,
                           'description': 'biased correlated random walk toward the nearest habitat',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:max_orient_distance': {'type': 'float', 'default': 0.0,'min': 0.0, 'max': 1.0e10, 'units': 'meters',
                           'description': 'maximum detection distance of the habitat for orientation',
                           'level': self.CONFIG_LEVEL_BASIC}})
        # life stages
        self._add_config({ 'biology:stage_puerulus': {'type': 'float', 'default': 0.0,'min': 0.0, 'max': 1.0e10, 'units': 'seconds',
                           'description': 'minimum age for transition to puerulus larvae',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:late_stage_phyllosoma': {'type': 'float', 'default': 0.0,'min': 0.0, 'max': 1.0e10, 'units': 'seconds',
                           'description': 'minimum age age of late stage phyllosoma',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:mid_stage_phyllosoma': {'type': 'float', 'default': 0.0,'min': 0.0, 'max': 1.0e10, 'units': 'seconds',
                           'description': 'minimum age age of middle stage phyllosoma',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:max_swimming_speed_puerulus': {'type': 'float', 'default': 30.0,'min': 0.0, 'max': 1.0e10, 'units': 'centimeters.seconds^-1',
                           'description': 'maximum swimming speed of the puerulus when cruising toward the habitat',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:min_swimming_speed_puerulus': {'type': 'float', 'default': 10.0,'min': 0.0, 'max': 1.0e10, 'units': 'centimeters.seconds^-1',
                           'description': 'minimum swimming speed of the puerulus when cruising toward the habitat',
                           'level': self.CONFIG_LEVEL_BASIC}})
        # vertical motion, as in pelagicplankton_moana.py
        self._add_config({ 'biology:vertical_position_daytime': {'type': 'float', 'default': -5.00,'min': -100.0, 'max':0.0, 'units': 'meters negative down',
                           'description': 'the depth a species is expected to inhabit during the day time, in meters, negative down',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:vertical_position_nighttime': {'type': 'float', 'default': -1.00,'min': -100.0, 'max':0.0, 'units': 'meters negative down',
                           'description': 'the depth a species is expected to inhabit during the night time, in meters, negative down',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:vertical_migration_speed_constant': {'type': 'float', 'default': 1.0e-4,'min': 0.0, 'max': 1.0, 'units': 'm/s',
                           'description': 'Constant vertical migration rate (m/s) towards vertical_position_daytime and vertical_position_nighttime levels',
                           'level': self.CONFIG_LEVEL_BASIC}})
        self._add_config({ 'biology:maximum_larvae_depth': {'type': 'float', 'default': -100.0,'min': -10000.0, 'max': -1.0, 'units': 'm',
                           'description': 'maximum depth the larvae can swim down to, larvae will swim up if reached',
                           'level': self.CONFIG_LEVEL_BASIC}})                       

#####################################################################################################################
# Accessory functions
#####################################################################################################################
    
    # Haversine formula to compute angles during orientation
    def haversine_angle(self, lon1, lat1, lon2, lat2):
        rlat1 = np.deg2rad(lat1)
        rlat2 = np.deg2rad(lat2)
        rlon1 = np.deg2rad(lon1)
        rlon2 = np.deg2rad(lon2)
        X = np.cos(rlat2)*np.sin(rlon2-rlon1)
        Y = np.cos(rlat1)*np.sin(rlat2)-np.sin(rlat1)*np.cos(rlat2)*np.cos(rlon2-rlon1)
        return np.arctan2(Y,X)
     
    def calculateMaxSunLight(self):
        # Calculates the max sun radiation at given positions and dates (and returns zero for night time)
        # 
        # The method is using the third party library PySolar : https://pysolar.readthedocs.io/en/latest/#
        # 
        # 
        # some other available options:
        # https://pypi.org/project/solarpy/
        # https://github.com/trondkr/pyibm/blob/master/light.py
        # use calclight from Kino Module here  : https://github.com/trondkr/KINO-ROMS/tree/master/Romagnoni-2019-OpenDrift/kino
        # ERcore : dawn and sunset times : https://github.com/metocean/ercore/blob/ercore_opensrc/ercore/lib/suncalc.py
        # https://nasa-develop.github.io/dnppy/modules/solar.html#examples
        # 
        from pysolar import solar
        date = self.time
        date = date.replace(tzinfo=timezone.utc) # make the datetime object aware of timezone, set to UTC
        logger.debug('Assuming UTC time for solar calculations')
        # longitude convention in pysolar, consistent with Opendrift : negative reckoning west from prime meridian in Greenwich, England
        # the particle longitude should be converted to the convention [-180,180] if that is not the case
        sun_altitude = solar.get_altitude(self.elements.lat, self.elements.lon, date) # get sun altitude in degrees
        sun_azimut = solar.get_azimuth(self.elements.lat, self.elements.lon, date) # get sun azimuth in degrees
        sun_radiation = np.zeros(len(sun_azimut))
        # not ideal get_radiation_direct doesnt accept arrays...
        for elem_i,alt in enumerate(sun_altitude):
            sun_radiation[elem_i] = solar.radiation.get_radiation_direct(date, alt)  # watts per square meter [W/m2] for that time of day
        # save compute light for each particle 
        self.elements.light = sun_radiation * 4.6 #Converted from W/m2 to umol/m2/s-1"" - 1 W/m2 ≈ 4.6 μmole.m2/s
        logger.debug('Solar radiation from %s to %s [W/m2]' % (sun_radiation.min(), sun_radiation.max() ) )
        # print(np.min(sun_radiation))
        # print(date)


    def sea_surface_height(self):
        '''fetches sea surface height for presently active elements
           sea_surface_height > 0 above mean sea level
           sea_surface_height < 0 below mean sea level
        '''
        if hasattr(self, 'environment') and \
                hasattr(self.environment, 'sea_surface_height'):
            if len(self.environment.sea_surface_height) == \
                    self.num_elements_active():
                sea_surface_height = \
                    self.environment.sea_surface_height
        if 'sea_surface_height' not in locals():
            env, env_profiles, missing = \
                self.get_environment(['sea_surface_height'],
                                     time=self.time, lon=self.elements.lon,
                                     lat=self.elements.lat,
                                     z=0*self.elements.lon, profiles=None)
            sea_surface_height = \
                env['sea_surface_height'].astype('float32') 
        return sea_surface_height
    
    def surface_stick(self):
        '''Keep particles just below the surface.
           (overloads the OpenDrift3DSimulation version to allow for possibly time-varying
           sea_surface_height)
        '''
        
        sea_surface_height = self.sea_surface_height() # returns surface elevation at particle positions (>0 above msl, <0 below msl)
        
        # keep particle just below sea_surface_height (self.elements.z depth are negative down)
        surface = np.where(self.elements.z >= sea_surface_height)
        if len(surface[0]) > 0:
            self.elements.z[surface] = sea_surface_height[surface] -0.01 # set particle z at 0.01m below sea_surface_height


    def get_circle(self,centerLon,centerLat,radius_meters):
        # https://stackoverflow.com/questions/15886846/python-elegant-way-of-finding-the-gps-coordinates-of-a-circle-around-a-certain
        radius_earth_meters = 6378137 
        angle = np.linspace(0,2*np.pi,36)
        dx = radius_meters * np.cos(angle)
        dy = radius_meters * np.sin(angle)
        lat_circle = centerLat + (180 / np.pi) * (dy / radius_earth_meters)
        lon_circle = centerLon + (180 / np.pi) * (dx / radius_earth_meters) / np.cos(centerLat * np.pi / 180)
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.plot(centerLon,centerLat,'r.')
        # plt.plot(lon_circle,lat_circle,'g.')
        # import pdb;pdb.set_trace()
        return lon_circle ,lat_circle

        # see also..but slower
        # https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python/289923
        # 
        # from functools import partial
        # import pyproj
        # from shapely.ops import transform
        # from shapely.geometry import Point
        #  proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')
        # # Azimuthal equidistant projection
        # local_azimuthal_projection = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
        # project = partial(pyproj.transform,
        #                   pyproj.Proj(local_azimuthal_projection.format(lat=centerLat, lon=centerLon)),
        #                   proj_wgs84)
        # buf = Point(0, 0).buffer(radius_meters)  # distance in metres
        # circle = transform(project, buf).exterior.xy
        # lon_circle = circle[0]
        # lat_circle = circle[1] 



#####################################################################################################################
# Definition of habitat
#####################################################################################################################
      
    def habitat(self, shapefile_location):
        """Suitable habitat in a shapefile"""
        polyShp = fiona.open(shapefile_location) # import shapefile
        polyList = []
        self.centers_habitat = []
        rad_centers = []
        for poly in polyShp: # create individual polygons from shapefile
            polyGeom = Polygon(poly['geometry']['coordinates'][0])
            polyList.append(polyGeom) # Compile polygon in a list 
            self.centers_habitat.append(polyGeom.centroid.coords[0]) # Compute centroid and return a [lon, lat] list
        for poly in range(len(self.centers_habitat)):
            rad_centers.append([np.deg2rad(self.centers_habitat[poly][1]),np.deg2rad(self.centers_habitat[poly][0])])
        self.multiShp = MultiPolygon(polyList).buffer(0) # Aggregate polygons in a MultiPolygon object and buffer to fuse polygons and remove errors
        self.ball_centers = BallTree(rad_centers, metric='haversine') # Create a Ball Tree with the centroids for faster computation
        return self.multiShp, self.ball_centers, self.centers_habitat


    def find_nearest_habitat(self,lon,lat):
        return self.ball_centers.query(list(zip(np.deg2rad(lat), np.deg2rad(lon))), k=1)
    
    
#####################################################################################################################
# Interaction with environment
#####################################################################################################################

    def interact_with_seafloor(self):
        """Seafloor interaction according to configuration setting"""
        # 
        # This function will overloads the version in basemodel.py
        if self.num_elements_active() == 0:
            return
        if 'sea_floor_depth_below_sea_level' not in self.priority_list:
            return
        sea_floor_depth = self.sea_floor_depth()
        below = np.where(self.elements.z < -sea_floor_depth)[0]
        if len(below) == 0:
                logger.debug('No elements hit seafloor.')
                return

        below_and_older = np.logical_and(self.elements.z < -sea_floor_depth, 
            self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds'))
        below_and_younger = np.logical_and(self.elements.z < -sea_floor_depth, 
            self.elements.age_seconds < self.get_config('biology:min_settlement_age_seconds'))
        
        # Move all elements younger back to seafloor 
        # (could rather be moved back to previous if relevant? )
        self.elements.z[np.where(below_and_younger)] = -sea_floor_depth[np.where(below_and_younger)]

        # deactivate elements that were both below and older
        if self.get_config('biology:settlement_in_habitat') is False:
            self.deactivate_elements(below_and_older ,reason='settled_on_bottom')
        # if elements can only settle in habitat then they are moved back to seafloor
        else:
            self.elements.z[np.where(below_and_older)] = -sea_floor_depth[np.where(below_and_older)]

        logger.debug('%s elements hit seafloor, %s were older than %s sec. and deactivated, %s were lifted back to seafloor' \
            % (len(below),len(below_and_older),self.get_config('biology:min_settlement_age_seconds'),len(below_and_younger)))    

    
    def interact_with_coastline(self,final = False): 
        """Coastline interaction according to configuration setting
           (overloads the interact_with_coastline() from basemodel.py)
           
           The method checks for age of particles that intersected coastlines:
             if age_particle < min_settlement_age_seconds : move larvaes back to previous wet position
             if age_particle > min_settlement_age_seconds : larvaes become stranded and will be deactivated.
        """
        i = self.get_config('general:coastline_action') # will always be 'previous'

        if not hasattr(self.environment, 'land_binary_mask'):
            return

        if final is True:  # Get land_binary_mask for final location
            en, en_prof, missing = \
                self.get_environment(['land_binary_mask'],
                                     self.time,
                                     self.elements.lon,
                                     self.elements.lat,
                                     self.elements.z,
                                     None)
            self.environment.land_binary_mask = en.land_binary_mask

        # if i == 'previous':  # Go back to previous position (in water)
        # previous_position_if = self.previous_position_if()
        if self.newly_seeded_IDs is not None:
                self.deactivate_elements(
                    (self.environment.land_binary_mask == 1) &
                    (self.elements.age_seconds == self.time_step.total_seconds()),
                    reason='seeded_on_land')
        on_land = np.where(self.environment.land_binary_mask == 1)[0]

            # if previous_position_if is not None:
            #     self.deactivate_elements((previous_position_if*1 == 1) & (
            #                      self.environment.land_binary_mask == 0),
            #                          reason='seeded_at_nodata_position')

        # if previous_position_if is None:
        #     on_land = np.where(self.environment.land_binary_mask == 1)[0]
        # else:
        #     on_land = np.where((self.environment.land_binary_mask == 1) |
        #                        (previous_position_if == 1))[0]
        if len(on_land) == 0:
            logger.debug('No elements hit coastline.')
        else:
            if self.get_config('biology:settlement_in_habitat') is True:
                    # Particle can only settle in habitat, set back to previous location
                    logger.debug('%s elements hit coastline, '
                              'moving back to water' % len(on_land))
                    on_land_ID = self.elements.ID[on_land]
                    self.elements.lon[on_land] = \
                        np.copy(self.previous_lon[on_land_ID - 1])
                    self.elements.lat[on_land] = \
                        np.copy(self.previous_lat[on_land_ID - 1])
                    self.environment.land_binary_mask[on_land] = 0  
            elif self.get_config('biology:min_settlement_age_seconds') == 0.0 :
                # No minimum age input, set back to previous position (same as in interact_with_coastline() from basemodel.py)
                logger.debug('%s elements hit coastline, '
                          'moving back to water' % len(on_land))
                on_land_ID = self.elements.ID[on_land]
                self.elements.lon[on_land] = \
                    np.copy(self.previous_lon[on_land_ID - 1])
                self.elements.lat[on_land] = \
                    np.copy(self.previous_lat[on_land_ID - 1])
                self.environment.land_binary_mask[on_land] = 0
            else:
                #################################
                # Minimum age before settling was input; check age of particle versus min_settlement_age_seconds
                # and strand or recirculate accordingly
                on_land_and_younger = np.where((self.environment.land_binary_mask == 1) & (self.elements.age_seconds < self.get_config('biology:min_settlement_age_seconds')))[0]
                on_land_and_older = np.where((self.environment.land_binary_mask == 1) & (self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds')))[0]

                # this step replicates what is done is original code, but accounting for particle age. It seems necessary 
                # to have an array of ID, rather than directly indexing using the "np.where-type" index (in dint64)
                on_land_and_younger_ID = self.elements.ID[on_land_and_younger] 
                on_land_and_older_ID = self.elements.ID[on_land_and_older]

                logger.debug('%s elements hit coastline' % len(on_land))
                logger.debug('moving %s elements younger than min_settlement_age_seconds back to previous water position' % len(on_land_and_younger))
                logger.debug('%s elements older than min_settlement_age_seconds remain stranded on coast' % len(on_land_and_younger))
                
                # refloat elements younger than min_settlement_age back to previous position(s)
                if len(on_land_and_younger) > 0 :
                    # self.elements.lon[np.where(on_land_and_younger)] = np.copy(self.previous_lon[np.where(on_land_and_younger)])  
                    # self.elements.lat[np.where(on_land_and_younger)] = np.copy(self.previous_lat[np.where(on_land_and_younger)])
                    # self.environment.land_binary_mask[on_land_and_younger] = 0 

                    self.elements.lon[on_land_and_younger] = np.copy(self.previous_lon[on_land_and_younger_ID - 1])
                    self.elements.lat[on_land_and_younger] = np.copy(self.previous_lat[on_land_and_younger_ID - 1])
                    self.environment.land_binary_mask[on_land_and_younger] = 0

                # deactivate elements older than min_settlement_age & save position
                # ** function expects an array of size consistent with self.elements.lon
                self.deactivate_elements((self.environment.land_binary_mask == 1) & \
                                         (self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds')),
                                         reason='settled_on_coast')
                    
    def interact_with_habitat(self):
           """Habitat interaction according to configuration setting
               The method checks if a particle is within the limit of an habitat before allowing settlement
           """        
           # Get age of particle
           old_enough = np.where(self.elements.age_seconds >= self.get_config('biology:min_settlement_age_seconds'))[0]
           # Extract particles positions
           if len(old_enough) > 0 :
               pts_lon = self.elements.lon[old_enough]
               pts_lat = self.elements.lat[old_enough]
               ## Check if position of particle is within boundaries of polygons => slow version because loop over polygons
               for i in range(len(pts_lon)): # => faster version
                    pt = Point(pts_lon[i], pts_lat[i])
                    in_habitat = pt.within(self.multiShp)
                    if in_habitat == True:
                        self.environment.land_binary_mask[old_enough[i]] = 6
           # Deactivate elements that are within a polygon and old enough to settle
           # ** function expects an array of size consistent with self.elements.lon                
           self.deactivate_elements((self.environment.land_binary_mask == 6), reason='settled_on_habitat')



#####################################################################################################################
# IBM: horizontal and vertical movements
#####################################################################################################################

    def reset_horizontal_swimming(self):
            # Create a  vector for swimming movement
            self.u_swim = np.array([0.0]*len(self.elements.lat))
            self.v_swim = np.array([0.0]*len(self.elements.lon))
            return self.u_swim, self.v_swim

    def direct_orientation_habitat(self):
            """"
            Direct orientation, where the larvae swim toward the nearest reef. 
            Biased correlated random walk toward the nearest habitat. 
            Equations described in Codling et al., 2004 and Staaterman et al., 2012
            ** Algorithm could be vectorized to optimize speed (instead of looping through each particle)
            """
            # Replaced by metamorphosis step where larvae have a transformation step
            # Check if the particles are old enough to orient
            # old_enough = np.where(self.elements.age_seconds >= self.get_config('biology:stage_puerulus'))[0]
            #
            # Check if particles are ready to metamorphose in puerulus
            puerulus = np.where(self.elements.metamorphosis == 1)[0]
            logger.debug('Larvae : direct orientation of puerulus - %s particles ready for orientation ' % (len(puerulus)))
            if len(puerulus) > 0 :
                # find closest habitat distance, and its index
                habitat_near, habitat_id = self.find_nearest_habitat(self.elements.lon[puerulus],self.elements.lat[puerulus])
                # old_close_enough = np.where( (self.elements.age_seconds >= self.get_config('biology:age_beginning_orientation')) & \
                #                              (habitat_near.ravel()*6371. < self.get_config('biology:max_orient_distance')) )[0]
                close_enough = (habitat_near.ravel()*6371. < self.get_config('biology:max_orient_distance')) # 6371km is earth radius 
                logger.debug('Larvae : direct orientation of puerulus - moving %s particles towards nearest reef' % (close_enough.sum()))

                #cur_dir_rad = self.get_current_direction() # not used for the lobster larvae

                # looping through each particle - could be vectorized
                for i in range(len(self.elements.lat[puerulus])):
                    if habitat_near[i][0]*6371 > self.get_config('biology:max_orient_distance'):
                        pass
                    else:
                        pt_lon = self.elements.lon[puerulus][i]
                        pt_lat = self.elements.lat[puerulus][i]
                        pt_lon_old = self.previous_lon[puerulus][i]
                        pt_lat_old = self.previous_lat[puerulus][i]
                        # Case where particle is old enough and close enough from habitat to orient
                        # Strength of orientation (depend on distance to the habitat). Eq. 3 Staaterman et al., 2012
                        dist = 1 - (habitat_near[i][0]*6371/self.get_config('biology:max_orient_distance')) # 
                        # Compute direction of nearest habitat. See Staaterman et al., 2012
                        theta_pref = - self.haversine_angle(pt_lon, pt_lat, self.centers_habitat[habitat_id[i][0]][0], self.centers_habitat[habitat_id[i][0]][1]) 
                        # Compute current direction from previous timestep
                        # ** note this will include the swimming-induced motions from past timestep as well
                        theta_current = self.haversine_angle(pt_lon_old, pt_lat_old, pt_lon, pt_lat) # from previous positions 
                        # theta_current = cur_dir_rad[old_enough[i]] # from ambient current direction
                        # Mean turning angle, Eq. 2 Staaterman et al., 2012
                        mu = -dist * (theta_current - theta_pref)
                        # Define larvae swimming direction
                        # direction uncertainty drawn from Von Mises distribution : First parameter: mu, second parameter: kappa
                        ti  = np.random.vonmises(0, 5) #  (control the uncertainty of orientation)
                        # final larvae swimming direction
                        theta = ti - theta_current - mu
                        # Compute u and v swim velocity component
                        self.elements.u_swim[puerulus[i]] = self.swimming_speed()*np.cos(theta)
                        self.elements.v_swim[puerulus[i]] = self.swimming_speed()*np.sin(theta)

            logger.debug('    %.5f [m/s] =< u_swim <=  %.5f [m/s]' % (np.min(self.elements.u_swim),np.max(self.elements.u_swim))) 
            logger.debug('    %.5f [m/s] =< v_swim <=  %.5f [m/s]' % (np.min(self.elements.u_swim),np.max(self.elements.u_swim))) 
            self.update_positions(self.elements.u_swim , self.elements.v_swim)

    def swimming_speed(self):
            # Compute swimming speed of larvae - no dependence on age 
            # beta distribution to reproduce the findings in Jeffs and Hollands 2000 (speeds from 13 and 22cm/s with outlier at 30.7cm/s) 
            # and Wilkin and Jeffs 2011 (model predictions btwn 13 and 16cm/s) (mean = 1/(1+B/A) => 16.8cm/s here with a=2 and b=8)
            # Not age dependent because puerulus do not develop further until they reach an habitat
            # swim_speed = (Vmin + random.beta*(Vmax-Vmin) )
            # 
            # Jeffs and Hollands,2000,Swimming behaviour of the puerulus of the spiny lobster, Jasus edwardsii.Crustaceana 73(7):847-856
            Vmin = self.get_config('biology:min_swimming_speed_puerulus')
            Vmax = self.get_config('biology:max_swimming_speed_puerulus')
            swimming_speed = ( Vmin + np.random.beta(2, 8) * (Vmax-Vmin) ) / 100
            return swimming_speed
   
    def get_current_direction(self):
        ''' returns current direction in the trigonometric convention, as used in direct_orientation_habitat'''
        uu = self.environment['x_sea_water_velocity']
        vv = self.environment['y_sea_water_velocity']
        return np.arctan2(vv,uu)

    
    def update_terminal_velocity(self,Tprofiles=None, Sprofiles=None,z_index=None): 
            ''' Diel vertical migration for late life stages (VIII to XI) only. Modified from pelagicplankton_moana.py developed by Simon Weppe
            Keep all stages within 100m of the surface '''
            
            # Modifies the same variable than update_terminal_velocity(), self.elements.terminal_velocity = W, but using a different algorithm.
            # Larvae are assumed to move to daytime or nighttime vertical positions in the water column, at a constant rate
            # the actual settling is taken care of in vertical_mixing() or vertical_buoyancy() (i.e. from OceanDrift methods)
            # it is expected that larve will go down during day time and up during night time but that is not fixed in the code. 
            # Particles will simply go towards the daytime or nighttime positions.
            # https://github.com/metocean/ercore/blob/ercore_nc/ercore/materials/biota.py#L80 
            vertical_velocity = np.abs(self.get_config('biology:vertical_migration_speed_constant'))  # magnitude in m/s 
            late_stage_phy = self.elements.age_seconds > self.get_config('biology:late_stage_phyllosoma') 
            logger.debug('Larvae : update_terminal_velocity - %s particles in late_stage_phyllosoma' % (late_stage_phy.sum()) )
            if late_stage_phy.any() > 0 :
                self.calculateMaxSunLight() # compute solar radiation at particle positions (using PySolar)
                z_day = self.get_config('biology:vertical_position_daytime')    #  the depth a species is expected to inhabit during the day time, in meters, negative down') #
                z_night = self.get_config('biology:vertical_position_nighttime') # 'the depth a species is expected to inhabit during the night time, in meters, negative down') #
                ind_day = np.where(late_stage_phy & (self.elements.light>0) )     #np.where(self.elements.light[late_stage_phy]>0)
                ind_night = np.where(late_stage_phy & (self.elements.light == 0) )  #np.where(self.elements.light[late_stage_phy]==0)
                logger.debug('Using constant migration rate (%s m/s) towards day and night time positions' % (vertical_velocity) )
                logger.debug('%s particles in day time' % (len(ind_day[0])))
                logger.debug('%s particles in night time' % (len(ind_night[0])))
                # for particles in daytime : particles below the daytime position need to go up while particles above the daytime position need to go down
                # (same for for particles in nighttime)
                # Note : depth convention is negative down in Opendrift
                # 
                # e.g. z=-5, z_day = -3, below daytime position,  need to go up (terminal_velocity>0) 
                #      diff = (z - z_day) = -2, so w = - np.sign(diff) * vertical_velocity
                self.elements.terminal_velocity[ind_day] = - np.sign(self.elements.z[ind_day] - z_day) * vertical_velocity
                self.elements.terminal_velocity[ind_night] = - np.sign(self.elements.z[ind_night] - z_night) * vertical_velocity
                # print(self.elements.z)
                
            # take into account maximum depth for other larval stages
            early_life = np.where(self.elements.age_seconds < self.get_config('biology:late_stage_phyllosoma'))[0]
            if len(early_life) > 0:
                logger.debug('%s particles deeper than max depth %s m, setting positive terminal_velocity' % ( len(early_life),self.get_config('biology:maximum_larvae_depth') ))
                too_deep = np.where(self.elements.z[early_life] < self.get_config('biology:maximum_larvae_depth') )[0]
                self.elements.terminal_velocity[too_deep] = np.abs(vertical_velocity) # set vertical swimming speed positive so particles go up, in m/s
  
  
###################################################################################################################
# Pelagic larval duration, transition to puerulus stage, and mortality
###################################################################################################################

    def phyllosoma_mortality(self):
        ''' Phyllosoma are not found inshore between 4 and 12 months of dispersal (i.e between mid and late Phyllosoma stages)
        This could be due to an increased in predatory pressure closer to the coast,
        therefore, we remove all the phyllosoma larvae that are found within 20km of the coast
        '''
        mid_stage_phyllosoma =  np.where( (self.elements.age_seconds >= self.get_config('biology:mid_stage_phyllosoma')) &	(self.elements.age_seconds <= self.get_config('biology:late_stage_phyllosoma')) )[0]
        logger.debug('Larvae : checking phyllosoma distance to shore - %s particles in mid to late_stage_phyllosoma' % (len(mid_stage_phyllosoma)))
        if len(mid_stage_phyllosoma) > 0:
            for i in range(len(self.elements.lon[mid_stage_phyllosoma])):
                pt_lon = self.elements.lon[mid_stage_phyllosoma][i]
                pt_lat = self.elements.lat[mid_stage_phyllosoma][i] 
                # Remove phyllosoma found 20km inshore, or any other habitats defined by user (habitat_near[i]*6371 = distance in km)
                lon_circle ,lat_circle = self.get_circle(pt_lon, pt_lat, 20e3) 
                # check if any land points within the 20km radius
                on_land, prof, missing = self.get_environment(['land_binary_mask'], self.time, \
                                                            np.array(lon_circle), np.array(lat_circle), \
                                                            0.0*np.array(lat_circle), None)
                if np.array(on_land.view('f')).any():
                    self.environment.land_binary_mask[mid_stage_phyllosoma[i]] = 8
        if (self.environment.land_binary_mask == 8).any():
            logger.debug('Larvae : removing %s phyllosoma that swam too close to shore' % (self.environment.land_binary_mask == 8).sum() )
            self.deactivate_elements((self.environment.land_binary_mask == 8), reason='swam_too_close_to_shore')

    
    def puerulus_transition(self):
            '''Transition between the phyllosoma and puerelus stage: possible after 12 months of dispersal.'''
            proba_transition = {11:0.0393700787401575, 12:0.118110236220472, 1:0.204724409448819, 2:0.283464566929134, 3:0.370078740157480, 4:0.460629921259843, 5:0.582677165354331, 6:0.759842519685039, 7:0.877952755905512, 8:0.944881889763779, 9:0.988188976377953, 10:1.0}
            #Dictionary of values of cumulative probability of settlement per month: Nov to Oct
            proba_month = proba_transition[self.time.month]
            random_proba = np.random.rand(len(self.elements.lat))
            # Check is particles are old enough to transition
            transform = np.where( (self.elements.age_seconds >= self.get_config('biology:stage_puerulus')) & (random_proba < proba_month))[0]
            if len(transform) > 0:
                self.elements.metamorphosis[transform] = 1


    def increase_age_and_retire(self):  # ##So that if max_age_seconds is exceeded particle is flagged as died
            """Increase age of elements, and retire if older than config setting.
               >essentially same as increase_age_and_retire() from basemodel.py, 
               only using a different reason for retiring particles ('died' instead of 'retired')
               .. could probably be removed...
            """
            # Increase age of elements
            self.elements.age_seconds += self.time_step.total_seconds()

            # Deactivate elements that exceed a certain age
            if self.get_config('drift:max_age_seconds') is not None:
                self.deactivate_elements(self.elements.age_seconds >=
                                         self.get_config('drift:max_age_seconds'),
                                         reason='died')

            # Deacticate any elements outside validity domain set by user
            if self.validity_domain is not None:
                W, E, S, N = self.validity_domain
                if W is not None:
                    self.deactivate_elements(self.elements.lon < W, reason='outside')
                if E is not None:
                    self.deactivate_elements(self.elements.lon > E, reason='outside')
                if S is not None:
                    self.deactivate_elements(self.elements.lat < S, reason='outside')
                if N is not None:
                    self.deactivate_elements(self.elements.lat > N, reason='outside')


###################################################################################################################
# Update position of the larvae
###################################################################################################################    

    def update(self):
        """Update positions and properties of elements."""

        # Simply move particles with ambient current
        self.advect_ocean_current()

        # Horizontal swimming: Puerulus stage swimming to shore
        if self.get_config('biology:direct_orientation_habitat') is True:
            self.puerulus_transition()       
            self.reset_horizontal_swimming()
            self.direct_orientation_habitat() # self.update_positions in the function direct_orientation_habitat

        if False:    
            # Advect particles due to surface wind drag,
            # according to element property wind_drift_factor
            self.advect_wind()
            # Stokes drift
            self.stokes_drift()

        # Vertical advection
        self.vertical_advection()
        # Turbulent Mixing
        if self.get_config('drift:vertical_mixing') is True:
            self.update_terminal_velocity()
            self.vertical_mixing()
        else:  # Buoyancy
            self.update_terminal_velocity()
            self.vertical_buoyancy()

        ## Settlement in habitat
        if self.get_config('biology:settlement_in_habitat') is True:
            self.interact_with_habitat()

        # Mortality due to shore proximity (<20km)
        # slow, may be more relevant to apply in post-processing 
        if True: 
            self.phyllosoma_mortality()
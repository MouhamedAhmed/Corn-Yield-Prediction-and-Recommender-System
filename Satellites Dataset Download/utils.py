import numpy as np
import ee

import os
import time

import pyproj
from shapely.geometry import Point, mapping
from functools import partial
from shapely.ops import transform


# helper def to extract the QA bits
def getQABits(image, start, end, newName):
  '''
  a function to extract the QA bits in a specific range of bits from the ImageCollection (in case of MODIS/061/MOD09A1)
  Inputs:
      - image:         QA band of an ee.Image to extract bits from
      - start:         start bit of interest in the QA flag
      - end:           end bit of interest in the QA flag 
      - newName:       the new name of the selected bits image
  '''
  # Compute the bits we need to extract.
  pattern = 0
  for i in range(start, end):
      pattern += 2**i
  
  # Return a single band image of the extracted QA bits, giving the band a new name.
  return image.select([0], [newName])\
                .bitwiseAnd(pattern)\
                .rightShift(start)


###################################################################################################################################################


# A def to mask out cloudy pixels.
def maskCloud(image):
  '''
  a function to mask clouds from an MODIS/061/MOD09A1 Image
  Inputs:
      - image:         ee.Image from MODIS/061/MOD09A1 to mask clouds from
  '''
  # Select the QA band.
  QA = image.select('StateQA')
  # Get the internal_cloud_algorithm_flag bit.
  internalQuality = getQABits(QA, 8, 13, 'internal_quality_flag')
  # Return an image masking out cloudy areas (i.e. if the selected bits are all zeros).
  return image.updateMask(internalQuality.eq(0))


###################################################################################################################################################


def norm_band(imgcoll, band, polygon):
  '''
  a function to normalize a band of ee.ImageCollection using this formula ((x-min)/(max-min)), then re-scale it to be (0~255)
  Inputs:
      - imgcoll:        ee.ImageCollection to normalize the band from
      - band:           the band name to be normalized (in-place)
      - polygon:        ee.Geometry.Polygon to be exported  
  '''

  # select the band to be normalized
  colIm = imgcoll.select(band)

  # get the min/max values of the band in the image collection 
  minImage=colIm.min()
  maxImage=colIm.max()
  maxValue = ee.Number(maxImage.reduceRegion(**{
    'reducer': ee.Reducer.max(),
    'geometry': polygon,
    'maxPixels':10e10,
    'scale': 500
    }).get(band))
  minValue = ee.Number(minImage.reduceRegion(**{
    'reducer' : ee.Reducer.min(),
    'geometry' : polygon,
    'maxPixels' : 10e10,
    'scale' : 500
    }).get(band))

  # normalize the band (0~255)
  colIm_norm = colIm.map(lambda image: ((image.subtract(minValue)).divide(maxValue.subtract(minValue))).multiply(256).uint8())

  # overwrite the same band in the image collection 
  imgcoll = imgcoll.combine(colIm_norm, overwrite=True)	

  return imgcoll


###################################################################################################################################################


def create_export_vid_bands_task(vid_collection, folder_name, polygon, vid_name):
  '''
  a function to export 3 bands of ee.ImageCollection of a specific polygon to a video

  Inputs:
      - vid_collection: ee.ImageCollection to export from
      - folder_name:    folder to save videos in, must be placed in /content/drive/MyDrive
      - polygon:        ee.Geometry.Polygon to be exported (lat/long system)
  '''
  # the export task on the UINT8 version of the video collection to be exported in folder_name folder with a scale of 100 (i.e. 100 meters per pixel)
  task = ee.batch.Export.video.toDrive(**{
      'collection': vid_collection.map(lambda img: img.uint8()),
      'description': vid_name,
      # 'dimensions': 720,
      'scale': 500,
      'folder':folder_name,
      'framesPerSecond': 12,
      'region': polygon,
      'crs':'EPSG:4326',
  })

  return task


###################################################################################################################################################


def create_export_tasks_for_all_bands(imgcoll, bands, folder_name, polygon, loc_id, lat, lon, start_date, end_date):
  '''
  a function to export all bands as videos

  Inputs:
      - imgcoll:        ee.ImageCollection to export from
      - bands:          a list of the names of the 3 bands to be exported as a video
      - folder_name:    folder to save videos in, must be placed in /content/drive/MyDrive
      - polygon:        ee.Geometry.Polygon to be exported (lat/long system)
      - lat:            latitude of the center
      - lon:            longitude of the center
      - start_date:     start date
      - end_date:       end date
  '''
  # export each 3 bands as a video
  chunks = [bands[x:x+3] for x in range(0, len(bands), 3)]
  chunks[-1] = bands[-3:]
  tasks = []
  for bands_chunk in chunks:
    # select the 3 bands
    vid_collection = imgcoll.select(bands_chunk)

    # video name containing bands to be exported
    vid_name = 'bands:'+str(bands_chunk[0])+','+str(bands_chunk[1])+','+str(bands_chunk[2]) + ';loc:' + str(loc_id) + ';s:' + start_date + ';e:' + end_date
    vid_name = vid_name.replace("'", "")
    vid_name = vid_name.replace(".", "")

    if not os.path.exists(os.path.join('/content/drive/MyDrive/', folder_name, vid_name+'.mp4')):
      # export the video 
      task = create_export_vid_bands_task(vid_collection, folder_name, polygon, vid_name)  
      tasks.append(task)
    else:
      continue

  return tasks


###################################################################################################################################################


def start_task(task):
  task.start()
  time.sleep(0.5) 
  return task
 

###################################################################################################################################################


def start_multiple_tasks(tasks, n):
  '''
  a function to run multiple ee tasks at a time
  Inputs:
      - tasks: list of ee.to_drive tasks
      - n: number of tasks to be started
  '''
  to_be_started_tasks = tasks[:min(n, len(tasks))]
  remaining_tasks = tasks[min(n, len(tasks)):]
  
  for task in to_be_started_tasks: 
    # start the task and sleep 0.5 sec
    task = start_task(task)
  
  return to_be_started_tasks, remaining_tasks


###################################################################################################################################################


def create_circular_bb_polygon(center, area=100):
  '''
  a function to sample points from circular polygon around specific lat/long center with a specific radius
  Inputs:
      - center: tuple of center lat/long coords of the required polygon 
      - area: area of the circular polygon in KM-squared
  '''
  point = Point(center[1], center[0])

  local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

  wgs84_to_aeqd = partial(
      pyproj.transform,
      pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
      pyproj.Proj(local_azimuthal_projection),
  )

  aeqd_to_wgs84 = partial(
      pyproj.transform,
      pyproj.Proj(local_azimuthal_projection),
      pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
  )

  point_transformed = transform(wgs84_to_aeqd, point)

  buffer = point_transformed.buffer(area*1000)

  buffer_wgs84 = transform(aeqd_to_wgs84, buffer)
    
  return ee.Geometry.Polygon(list(mapping(buffer_wgs84)['coordinates'][0]))


###################################################################################################################################################


def get_loc_satellite_tasks(sat, bands, start_dates, end_dates, loc_id, folder_name, latitude, longitude, polygon_area=100):
    '''
    a function to get all tasks for a specific location and multiple periods
    Inputs:
        - sat:                          satellite code
        - bands:                        a list of the names of bands to be exported from the satellite database
        - start_dates:                  list of start dates
        - end_dates:                    list of end dates
        - loc_id:                       location index in the unique lat/lon dataframe to be used in video names
        - folder_name:                  folder to save videos in, must be placed in /content/drive/MyDrive
        - latitude:                     latitude of the center
        - longitude:                    longitude of the center
        - polygon_area:                 polygon area in KM^2
    '''
    # create polygon around the center
    polygon = create_circular_bb_polygon((latitude, longitude), area=polygon_area)

    # create folder for the satellite if not exists
    sat_folder_path = '/content/drive/MyDrive/' + folder_name
    if not os.path.exists(sat_folder_path):
        os.mkdir(sat_folder_path)

    tasks = []
    # loop on all periods
    for i in range(len(start_dates)):
        start_date = start_dates[i]
        end_date = end_dates[i]
        
        # image collection of the county in the needed period
        imgcoll = ee.ImageCollection(sat)\
                    .filterDate(start_date,end_date)\
                    .filterBounds(polygon)
        
        # mask clouds in MODIS/061/MOD09A1
        if sat == 'MODIS/061/MOD09A1':
            imgcoll = imgcoll.map(maskCloud)

        # mask all pixels out of polygon
        imgcoll = imgcoll.map(lambda image: image.updateMask(ee.Image.constant(1).clip(polygon).mask()))

        # normalize (0~255) for the video export
        for band in bands:
            imgcoll = norm_band(imgcoll, band, polygon)

        # convert to uint8
        imgcoll = imgcoll.map(lambda img: img.uint8())
        tasks += create_export_tasks_for_all_bands(imgcoll, bands, folder_name, polygon, loc_id, latitude, longitude, start_date, end_date)
        
    return tasks


###################################################################################################################################################


def get_loc_tasks(latitude, longitude, year, loc_id, sat, folder_name, bands, polygon_area=50):
    '''
    a function to get all tasks for a specific location and year
    Inputs:
        - latitude:                     latitude of the center
        - longitude:                    longitude of the center
        - year:                         year of interest to get imagery in from 1st of April to 30th of September
        - loc_id:                       location index in the unique lat/lon dataframe to be used in video names
        - sat:                          satellite code
        - folder_name:                  folder to save videos in, must be placed in /content/drive/MyDrive
        - bands:                        a list of the names of bands to be exported from the satellite database
        - polygon_area:                 polygon area in KM^2
    '''

    start_date = str(int(year)) + '-04-01'
    end_date = str(int(year)) + '-09-30'

    tasks = get_loc_satellite_tasks(
                                        sat, 
                                        bands, 
                                        [start_date], 
                                        [end_date], 
                                        loc_id,
                                        folder_name, 
                                        latitude, 
                                        longitude, 
                                        polygon_area=polygon_area
                                    )
    return tasks
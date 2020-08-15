# qgis_3_load_rasters_north_east
# This program loads the rasters in the NE of PA to make sure processing worked correctly.
# It will print which layers did not load.
from qgis.core import *
import qgis.utils

# get the path to the shapefile e.g. /home/project/data/ports.shp
path_to_layer = "/home/student/charcoalhearths/data_known_points/Charcoal Hearths_8-11-2020_bpc.shp"
  
vlayer = iface.addVectorLayer(path_to_layer, "Charcoal Hearths layer", "ogr")
if not vlayer:
    print("Layer failed to load!")

# load the predictions
path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200720T1614/4326_000_hearth_prediction_points.shp"
  
vlayer = iface.addVectorLayer(path_to_layer, "Charcoal Hearths predictions layer", "ogr")
if not vlayer:
    print("Layer failed to load!")

report=""
import csv

areas_nsew = {}
with open('/home/student/charcoalhearths/data_sheet_areas_nsew.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Number'],row['NSEW'])
        area = row['Number']
        area="0"+area
        area=area[-3:]
        nsew = row['NSEW']
        areas_nsew[area] = nsew

active_region = "NE"

#layer style
panSymbol = QgsFillSymbolV2.createSimple({'color':'0,0,0,0', 'color_border':'#00ff00', 'width_border':'0.3'})
pasSymbol = QgsFillSymbolV2.createSimple({'color':'0,0,0,0', 'color_border':'#0000ff', 'width_border':'0.3'})

with open('/home/student/charcoalhearths/data_sheet.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Area'])
        area=row['Area']
        area="0"+area
        area=area[-3:]
        if(areas_nsew[area] == active_region):
            pan=""
            pas=""
            pan = row['PAN']
            
            area_num = int(area)
            part_num = ""
            if(area_num > 0 and area_num < 98):
                part_num = "1"
            if(area_num > 97 and area_num < 174):
                part_num = "2"
            if(area_num > 173 and area_num < 252):
                part_num = "3"
            if(area_num > 251 and area_num < 336):
                part_num = "4"
                
            if(pan == "Y"):
                path_to_layer = "/storage/images/part_" + part_num + "/" + str(area_num) + "blast2demPAN_Slope_DEM.tif"
                
                vlayer = iface.addRasterLayer(path_to_layer, area+"pan layer")
                if not vlayer:
                    print(area+ "pan layer failed to load!")
                    report = report + "\n" + area+ "pan layer failed to load!"

            pas = row['PAS']
            if(pas == "Y"):
                path_to_layer = "/storage/images/part_" + part_num + "/" + str(area_num) + "blast2demPAS_Slope_DEM.tif"
                vlayer = iface.addVectorLayer(path_to_layer, area+"pas layer", "ogr")
                if not vlayer:
                    print(area+ "pas layer failed to load!")
                    report = report + "\n" + area+ "pas layer failed to load!"
print(report) 
# qgis_2_add_poly_layers_for_nort_east
# This program loads all of the area polys to make sure processing worked correctly.
# It will print which layers did not load.
from qgis.core import *
import qgis.utils

# get the path to the shapefile e.g. /home/project/data/ports.shp
path_to_layer = "/home/student/charcoalhearths/data_known_points/Charcoal Hearths_8-11-2020_bpc.shp"
   
vlayer = iface.addVectorLayer(path_to_layer, "Charcoal Hearths layer", "ogr")
if not vlayer:
    print("Layer failed to load!")

# State Game Lands
path_to_layer = "/home/student/charcoal_hearth_hill/state_game_lands/PGC_StateGamelan2018 WGS84_UTMzone18N.shp"

vlayer = iface.addVectorLayer(path_to_layer, "State Game Lands Boundaries layer", "ogr")
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
            if(pan == "Y"):
                path_to_layer = "/home/student/charcoal_hearth_hill/polys/"+area+"pan_tile_boundary.shp"
                vlayer = iface.addVectorLayer(path_to_layer, area+"pan layer", "ogr")
                if not vlayer:
                    print(area+ "pan layer failed to load!")
                    report = report + "\n" + area+ "pan layer failed to load!"
                else:
                    layerRenderer  = vlayer.rendererV2()
                    layerRenderer.setSymbol(panSymbol)
                    vlayer.triggerRepaint()
                #Predictions
                path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200720T1614/"+area+"pan_predictions.shp"
                vlayer = iface.addVectorLayer(path_to_layer, area+"pan layer", "ogr")
                if not vlayer:
                    print(area+ "pan_predictions layer failed to load!")
                    report = report + "\n" + area+ "pan_predictions layer failed to load!"


            pas = row['PAS']
            if(pas == "Y"):
                path_to_layer = "/home/student/charcoal_hearth_hill/polys/"+area+"pas_tile_boundary.shp"
                vlayer = iface.addVectorLayer(path_to_layer, area+"pas layer", "ogr")
                if not vlayer:
                    print(area+ "pas layer failed to load!")
                    report = report + "\n" + area+ "pas layer failed to load!"
                else:
                    layerRenderer  = vlayer.rendererV2()
                    layerRenderer.setSymbol(pasSymbol)
                    vlayer.triggerRepaint()
                #Predictions
                path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200720T1614/"+area+"pas_predictions.shp"
                vlayer = iface.addVectorLayer(path_to_layer, area+"pas layer", "ogr")
                if not vlayer:
                    print(area+ "pas_predictions layer failed to load!")
                    report = report + "\n" + area+ "pas_predictions layer failed to load!"                    

            #vlayer.setCustomProperty("labeling", "pal")
            #vlayer.setCustomProperty("labeling/enabled", "true")
            #vlayer.setCustomProperty("labeling/fontFamily", "Arial")
            #vlayer.setCustomProperty("labeling/fontSize", "10")
            #vlayer.setCustomProperty("labeling/fieldName", "layerName")
            #vlayer.setCustomProperty("labeling/placement", "2")
print(report) 
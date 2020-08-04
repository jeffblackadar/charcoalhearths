# qgis_1_check_predictions
from qgis.core import *
import qgis.utils

# get the path to the shapefile e.g. /home/project/data/ports.shp
path_to_layer = "/storage/images/charcoal_hearth_hill/downloads/Weston_Uploads/Shapefiles_5-20-2020/Charcoal-Hearths.shp"
#/storage/images/charcoal_hearth_hill/polys/
# The format is:
# vlayer = QgsVectorLayer(data_source, layer_name, provider_name)

   
vlayer = iface.addVectorLayer(path_to_layer, "Ports layer", "ogr")
if not vlayer:
    print("Layer failed to load!")

report=""
import csv
with open('/home/student/charcoalhearths/data_sheet.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Area'])
        area=row['Area']
        area="0"+area
        area=area[-3:]
        pan=""
        pas=""
        pan = row['PAN']
        if(pan == "Y"):
            path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200720T1614/"+area+"pan_predictions.shp"
            vlayer = iface.addVectorLayer(path_to_layer, area+"pan layer", "ogr")
            if not vlayer:
                print(area+ "pan layer failed to load!")
                report = report + "\n" + area+ "pan_predictions layer failed to load!"
        pas = row['PAS']
        if(pas == "Y"):
            path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200720T1614/"+area+"pan_predictions.shp"
            vlayer = iface.addVectorLayer(path_to_layer, area+"pas layer", "ogr")
            if not vlayer:
                print(area+ "pas_predictions layer failed to load!")
                report = report + "\n" + area+ "pas layer failed to load!

print(report)  

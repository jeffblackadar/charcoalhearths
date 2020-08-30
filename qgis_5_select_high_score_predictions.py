# qgis_1_check_predictions
from qgis.core import *
import qgis.utils
# get the path to the shapefile e.g. /home/project/data/ports.shp

path_to_layer = "/home/student/charcoal_hearth_hill/polys/cfg20200826T2315/4326_000_hearth_prediction_points.shp"
# The format is:
# vlayer = QgsVectorLayer(data_source, layer_name, provider_name)
vlayer_pred = iface.addVectorLayer(path_to_layer, "Predicted", "ogr")
if not vlayer_pred:
    print("Layer of predictions failed to load!")
    
path_to_layer = "/home/student/charcoalhearths/data_known_points/Charcoal Hearths_8-11-2020_bpc.shp"
# The format is:
# vlayer = QgsVectorLayer(data_source, layer_name, provider_name)
vlayer_known = iface.addVectorLayer(path_to_layer, "Known", "ogr")
if not vlayer_known:
    print("Layer failed to load!")

expression = 'score >= 0.8'
request = QgsFeatureRequest().setFilterExpression(expression)

matches = 0
for f in vlayer_pred.getFeatures(request):
   matches += 1
print(matches)
assert(matches == 43254)

    

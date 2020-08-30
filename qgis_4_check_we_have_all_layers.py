#Compare Games Lands from Wikipedia
from qgis.core import *
import qgis.utils

report="Not found: "
import csv

areas_project = {}
with open('/home/student/charcoalhearths/data_sheet.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Area'])
        area=row['Area']
        area="0"+area
        area=area[-3:]
        areas_project[area] = area


with open('/home/student/charcoalhearths/data_sheet_areas_nsew.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['Number'],row['NSEW'])
        area = row['Number']
        area="0"+area
        area=area[-3:]
        try:
            print(areas_project[area])
        except:
            print("not found",area)
            report = report + "," + area
print(report)

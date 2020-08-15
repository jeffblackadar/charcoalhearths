# charcoalhearths
Work with geotifs to detect charcoal hearths

Geotifs images are searched for objects resembling charcoal hearths.
To prepare of searching, download the files and split them into uniform sized rectangles.

Download geotifs from shared drive: 
All Data from Weston\LAS_Files\Part_[1-4]\[Area 52]QGISDEM\

Save to:
/storage/images/part_[1-4]/
example:
/storage/images/part_1/51blast2demPAS_Slope_DEM.tif

Run 0_1_check_that_all_DEM_Slope_tif_are_downloaded to verify all files are downloaded and to prepare to split them into rectangles.

In QGIS run
qgis_4_check_we_have_all_layers.py
This will check the files against a list of State Game Lands from Wikipedia. (An extra check)

## Points
12 August 2020 - This version of points is used for known charcoal hearths:
Charcoal_Hearths_8-11-2020_bpc
Update line 163
When this is updated, 0_split_tifs must all be run again.

![](http://jeffblackadar.ca/charcoal_hearths/splt_tifs_1_crop.png)
 


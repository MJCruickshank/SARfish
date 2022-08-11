# SARfish
Ship detection in Sentinel 1 Synthetic Aperture Radar (SAR) imagery

!["SARfish"](https://github.com/MJCruickshank/SARfish/blob/main/title_image.jpg)

*Note: This program is very much a work in progress, and its outputs should not be relied upon for important tasks.*

SARfish is a program designed to help Open Source Intelligence (OSINT) researchers investigate maritime traffic. While the automatic identification system (AIS) tracks most commercial vessels, a small percentage of vessels sail with their AIS transponders off. These vessels are often military vessels, or shipments of illicit/clandestine cargoes, making them of particular interest to researchers. 

The program runs on a Faster R-CNN model with a ResNet-50-FPN backbone retrained on the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0). It takes Sentinel-1 VH polarisation images as an input and outputs a geojson file with points where a ship has possibly been detected. 

###Installation

###Usage Steps

1) Download a Sentinel 1 SAR VH polarisation image from Copernicus Open Access Hub (https://scihub.copernicus.eu/) or SentinelHub EO Browser (https://apps.sentinel-hub.com/eo-browser/)
2) Convert raw .tiff image to .tif (Can be performed in QGIS)
3) Run: 
```shell
python SARfish.py input_tif_image_path image_shard_directory output_geojson_filepath
```
   Example: 
```shell
python SARfish.py /Users/Sar_Ship_Detection/test_imagery/VH_test_image.tif /UsersSar_Ship_Detection/test_imagery/shards/ /Users/Sar_Ship_Detection/test_imagery/detections.geojson
```
4) Plot detections / imagery in GIS software

###Known Issues

###Future Development

# SARfish
Ship detection in Sentinel 1 Synthetic Aperture Radar (SAR) imagery

!["SARfish"](https://github.com/MJCruickshank/SARfish/blob/main/title_image.jpg)

## Description

*Note: This program is very much a work in progress, and its outputs should not be relied upon for important tasks.*

SARfish is a program designed to help Open Source Intelligence (OSINT) researchers investigate maritime traffic. While the automatic identification system (AIS) tracks most commercial vessels, a small percentage of vessels sail with their AIS transponders off. These vessels are often military vessels, or shipments of illicit/clandestine cargoes, making them of particular interest to researchers. 

The program runs on a Faster R-CNN model with a ResNet-50-FPN backbone retrained on the Large-Scale SAR Ship Detection Dataset-v1.0 (LS-SSDD-v1.0). It takes Sentinel-1 VH polarisation images as an input and outputs a geojson file with points where a ship has possibly been detected. 

Specifically, SARfish breaks down the input SAR geotiff file into 800x800 shards. Each of these shards is converted to a .jpg image and the model searches it for detections. The x,y coordinates of the detections are then converted into lat/lon and added to a list, before the program moves onto the next shard. Once all detections have been performed, the coordinates of potential ship detections are then checked for intersection with a buffered map of global land areas, and given a True/False value based on this, allowing for onshore detections to be filtered out.  

## Getting Started

### Requirements

- **Python 3.9** 
- **conda**: The installation script that installs the dependencies needs to use both conda and pip to fetch the required 
dependencies, so please use conda and create a new conda virtual environment.

### Installing Package Dependencies

1. Create the conda environment. This will install all necessary package dependencies too.

```shell
conda env create -f environment.yml
```

2. Activate the conda environment created.

```shell
conda activate SARfish
```
## To Run

1) Download a Sentinel 1 SAR VH polarisation image, for more details check the [Data Specifics](#data-specifics) section below
2) Convert raw .tiff image to .tif (Can be performed in QGIS)
3) Clone this repository
4) Download model weights here (https://drive.google.com/file/d/1f4hJH9YBeTlNkbWUrbCP-C8ELh0eWJtT/view) and save the model.bin file to the SARfish directory.
5) Change working directory to that of this repository
6) Run: 
```shell
python SARfish.py input_tif_image_name output_geojson_filename
```
   Example: 
```shell
python SARfish.py VH_test_image.tif detections.geojson
```
7) Plot detections / imagery in GIS software. Use the "onshore_detection" field in the output geojson file to filter out erronous detections on land.

### Data Specifics
You can download Sentinel 1 products from [Copernicus Open Access Hub](https://scihub.copernicus.eu/) or 
[SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/). The pipeline currently expects the Sentinel tile 
to be in EPSG:4326, so either you download the tile in that coordinate system or you need to reproject it. 
The datatype of the tile should be an `8-bit` integer.

## Known Issues

Currently the model's detection threshold is set quite low. This can result in false positives where objects like stationary oil platforms, rocks, or small islands can be detected as ships. 

Areas on the edge of the input raster may not be properly scanned, due to the image not being perfectly divisible by the 800x800 detection window. 

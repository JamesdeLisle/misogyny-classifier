#!/bin/bash

export DATA_FILE_NAME=ManualTag_Misogyny.csv

mkdir data
cd data
wget https://data.mendeley.com/public-files/datasets/3jfwsdkryy/files/1424e5df-994a-4927-86c1-608fbc81d2a0/file_downloaded -O ${DATA_FILE_NAME}
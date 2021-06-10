# Face-recognition

## Requirements

* python3
* cmake
* virtualenv module and linux package (recommeded but not mandatory)

## Installing dependencies

```shell
pip3 install -r requirements.txt 
```
## Downloading weights

```shell
cd modules/model-weights
./get_models.sh
```

* Windows users can download it from the below link
_https://docs.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX_ then extract and move _yolov3-wider_16000.weights_ file to _modules/model-weights/model-weights_ folder.

## Starting the app

Navigate yourself to root directory of the project and then
```shell
python3 main.py
```

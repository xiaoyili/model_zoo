# Sample

Required packages: `tensorflow-gpu==1.14.0`, `numpy`, `opencv-python`

## How to convert raw data to tfrecords

1. Download raw CIFAR-10 data set from [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

2. Unzip the `.tar.gz` package to a location

3. In `data_conversion.py`, replace the glob search pattern in line 13 with the new file search pattern, with which the unzipped raw data can be found.

4. `python data_conversion.py` The generated tfrecords will be in the same folder as `data_conversion.py`, each should be around 13MB.

## How to train the model:

1. Make sure the required Python packages are installed

2. Generate tfrecords from raw data (see above), or download the generated tfrecords, located in the following OSS path:
```
oss://luci-hangzhou/jingyu/cifar-10_for_ali/
```

3. Run the script 
```shell script
python model.py --i /path/to/tfrecords/cifar10-?.tfrecord -o /path/of/the/model/to/save
```
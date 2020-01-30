# RGB stream to events conversion module

## Install

1. Install dependencies
```
sudo apt install libeigen3-dev cmake
``` 
2. Compile module
```
git submodule update --init --recursive
mkdir build && cd build && cmake -DPYTHON_EXECUTABLE:FILEPATH=`which python3` -DCMAKE_BUILD_TYPE=Releas .. && cmake --build . && cd ..
```

## Test

1. Install additional python packages
```
pip3 install -r requirements.txt
```

2. Test simulate DVS sensor by producing events between two input images frame0000.jpg and frame0001.jpg. It writes results to out.hdf5
```
python3 simulate.py -i1 data/frame0000.jpg -i2 data/frame0001.jpg -o res/out.hdf5
```

## Overview

The pydvssimulator module implements DVSSimulator class. Its constructor takes initial image in grayscale, corresponding timestamp and sensitivity of the sensor (float):
```
sim = DVSSimulator(init_image, init_timestamp, C)
```

The class implements method update that constructs a set of events between the previous and the current frame. It takes the current frame in grayscale and the corresponding timestamp.
```
events = sim.update(img, timestamp)
```

It produces events as a dictionary with keys `timestamps`, `x_positions`, `y_positions` and `polarities`.
Event components ha:
* `uint64_t` for timestamps
* `uint32_t` for x coordinates
* `uint32_t` for y coordinates
* `bool` for polarities

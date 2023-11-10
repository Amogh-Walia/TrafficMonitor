<div align="center">

# Traffic Monitoring using Yolov5 and Deep Sort

</div>


<div align="center">
![output](https://github.com/Amogh-Walia/TrafficMonitor/assets/72308844/691e85bc-f1be-4d5a-a6a2-769f388abe69)

</div>



### On CPU - `12 to 15 FPS` 

## Pre-requisites : 

1) Clone the Repository 

```bash
git clone https://github.com/Amogh-Walia/TrafficMonitor

cd vehicle-counting-yolov5
```

2) Clone the legacy Yolo-v5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
```
   
4) Install the libraries
```bash
pip install -r requirements.txt
```




## Directory Structure :

After completing the above steps your directory should look like somewhat as of below structure

- `vehicle-counting-yolov5`
   - deep_sort
   - yolov5
   - input.mp4
   - yolov5s.pt
   - tracker.py
   - requirements.txt

## Run the algorithm 

``` bash
python tracker.py 
# This will download model weight - yolov5s.pt to base folder on first execution.
```

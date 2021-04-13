# Training phase for NCDBAD

Training phase allows to :
- Generate profiling (CPU usage) information
- Compute confusion matrix for detection performance
- Serializable and save training model

## Packages required 
- xgboost
```
pip3 install xgboost
```
- numpy
```
pip3 install numpy
```
- scipy 
```
pip3 install scipy
```
- tensorflow
```
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv
pip3 install tensorflow
``` 
or
```
pip3 install --user --upgrade tensorflow
```
- keras 
```
pip3 install keras
```
- sklearn 
```
pip3 install sklearn
```
- pandas
```
pip3 install pandas
```
- numpy 
```
pip3 install numpy
```
- stuptools
```
pip3 install stuptools
```

## Run
- Use launch_scrip.sh with at command
```
at -f /path/launch_script.sh now + 1 min
```

```
at -f /home/serinell/i-sec-dataset/training_script/launch_script.sh now + 1 min
```

```
at -f /home/bserinelli/Documents/project/i-sec-dataset/training_script/launch_script_2.sh now + 1 min
```


```
at -f file time_expression
```
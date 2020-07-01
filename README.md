# CME-CNN
This is an implementation of the CNN regression model for Coronal Mass Ejections (CMEs) arrival time prediction using Python 3, Keras and TensorFlow. The model is trained to learn the mapping between features of CMEs white-light observations (from SOHO LASCO C2) and its corresponding transit time from the Sun to the Earth. For example, the features extracted by the first convolutional layer for a CME is shown below: <br />
![](https://github.com/yiminking/CME-CNN/blob/master/imgs/first_max_pooling_output.png)

**[CME Arrival Time Prediction Using Convolutional Neural Network](https://iopscience.iop.org/article/10.3847/1538-4357/ab2b3e/meta)** <br /> 
Yimin Wang, Jiajia Liu, Ye Jiang, Robert Erdélyi <br /> 
*The Astrophysical Journal*, 2019

## Model structure
The architecture of the proposed CNN regression model is: <br />
![](https://github.com/yiminking/CME-CNN/blob/master/imgs/cnn_model.png)

## Datasets
A catalogue of all observed geo-effective CMEs since the beginning of the SOHO era, i.e. from 1996 to early 2018 was established by combining the following four CME databases: the Richardson and Cane list (http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm), the full halo CME list provided by the University of Science and Technology of China (http://space.ustc.edu.cn/dreams/fhcmes/index.php), the George Mason University CME/ICME list (http://solar.gmu.edu/heliophysics/index.php/GMU_CME/ICME_List), and the CME Scoreboard by NASA (https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/). After removing duplicates, 276 geo-effective events were obtained. Deatails on the dataset construction could be found in the paper (see **citation** below).

As an example, the folder "data" contains a subset of the data used in this paper as an example.

## Training
To train a new model: <br />
```python3 train.py```

## Prediction analyses
To visualize the model: <br />
```python3 visualize_model.py``` <br />
<br />
To analyse prediction performance: <br />
```python3 results_analysis.py``` <br />

## Citation
If you use this code for your research, please cite our paper **CME Arrival Time Prediction Using Convolutional Neural Network**: <br />
```
@article{cme2019,
  title={CME Arrival Time Prediction Using Convolutional Neural Network},
  author={Yimin Wang, Jiajia Liu, Ye Jiang, Robert Erdélyi},
  journal={The Astrophysical Journal},
  year={2019}
}
```

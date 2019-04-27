# CME-CNN
This is an implementation of the CNN regression model for CME arrival time prediction on Python 3, Keras and TensorFlow. The model is trained to learn the mapping between features of CME image observations and its corresponding transit time between the Sun and the Earth. For example, the features extracted by the first convolutional layer for a CME is shown below: <br />

![](https://github.com/yiminking/CME-CNN/blob/master/imgs/first_max_pooling_output.png)

**CME Arrival Time Prediction Using Convolutional Neural Network** <br /> 
Yimin Wang, Jiajia Liu, Ye Jiang, Robert Erdélyi <br /> 
*The Astrophysical Journal*, 2019

The architecture of the proposed CNN regression model is: <br />

![](https://github.com/yiminking/CME-CNN/blob/master/imgs/cnn_model.png)

If you use this code for your research, please cite our paper **CME Arrival Time Prediction Using Convolutional Neural Network**: <br />
```
@article{cme2019,
  title={CME Arrival Time Prediction Using Convolutional Neural Network},
  author={Yimin Wang, Jiajia Liu, Ye Jiang, Robert Erdélyi},
  journal={The Astrophysical Journal},
  year={2019}
}
```

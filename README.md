# DRDC-PCP: Novel Class Discovery for Hyperspectral Image via Class-Relation Perceptive Distillation with Prototype-level Clustering Prediction 


### Abstract

Confronted with the increasing emergency of hyperspectral remote sensing categories in the dynamic environment, traditional classification models that depend on fixed-category labeled data encounter difficulties on new classes recognition. Novel class discovery (NCD) aims to discover unknown class-disjoint novel classes in an unlabeled dataset with the pre-existing knowledge of known classes. Notably, the critical goal of NCD is to ensure recognition accuracy of known classes while identifying new ones. In this paper, we propose a class-relation perceptive distillation with prototype-level clustering prediction network (CRPD-PCP) for NCD of hyperspectral image (HSI). The proposed framework comprises an initial training stage (ITS) and a novel class discovery stage (NCDS) with two essential modules. Specifically, we present the class relation perceptive distillation (CRPD) module, which imposes a similarity constraint on the prediction of the distribution of new class data over the models of two stages.  With the CRPD operated on the NCDS, our model effectively captures class relation information in spectral-spatial domain between known and novel classes of HSI to avoid forgetting old knowledge. Besides, we establish the prototype-level clustering prediction (PCP) module to generate high-confidence pseudo-labels for unlabeled novel classes. To be specific, we progressively cluster samples with the same spectral angular distance from the perspective of prototypes, and the self-supervised prototype-level knowledge distillation strategy in PCP facilitates effective identification of new categories. Experiments conducted on four datasets demonstrate that the CRPD-PCP model generates superior performance compared to other NCD methods for HSI.

### Platform

This code was developed and tested with pytorch version 1.12.1

### Setting

First you need to download a Houston, Botswana, Salinas, Pavia dataset and put it in the Dataset folder.

Then all data sets are sliced 9×9 to generate pre-processed intermediate files and placed in the Data folder and modify the model.py file and the Test.py file.

This code gives you a Houston data set run by default where you may want to modify the parameters to suit the data set.

### Train

```
$ python Train.py
```

### Test

```
$ python Test.py
```



### Result

You can view the saved model in the model_saved_check folder. View the test results in result.txt. y_pre_houston_9.mat file was drawn to realize the visualization of hyperspectral image prediction results.

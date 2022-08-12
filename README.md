# Reimplementing AlexNet using pyTorch and VOC 2007 dataset

As a part of Codable machine learning focus group, I implemented AlexNet using pyTorch where I used VOC 2007 dataset to train 20 different classes

## Optimized results

``` python
mAP_test, test_loss, test_aps = test_classifier(test_loader, classifier, criterion)
-------  Class: aeroplane        AP:   0.5838  -------
-------  Class: bicycle          AP:   0.3188  -------
-------  Class: bird             AP:   0.1850  -------
-------  Class: boat             AP:   0.2781  -------
-------  Class: bottle           AP:   0.1636  -------
-------  Class: bus              AP:   0.1634  -------
-------  Class: car              AP:   0.5641  -------
-------  Class: cat              AP:   0.2670  -------
-------  Class: chair            AP:   0.3212  -------
-------  Class: cow              AP:   0.1445  -------
-------  Class: diningtable      AP:   0.2726  -------
-------  Class: dog              AP:   0.2162  -------
-------  Class: horse            AP:   0.5707  -------
-------  Class: motorbike        AP:   0.3191  -------
-------  Class: person           AP:   0.7042  -------
-------  Class: pottedplant      AP:   0.1584  -------
-------  Class: sheep            AP:   0.1992  -------
-------  Class: sofa             AP:   0.2470  -------
-------  Class: train            AP:   0.4068  -------
-------  Class: tvmonitor        AP:   0.1684  -------
mAP: 0.3126
Avg loss: 0.20124676778912545
```

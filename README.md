# Lung-Tumor-Segmentation-Using-PyTorch-DeepLearning

First, we need to obtain and preprocess the data for the segmentation task
The data is provided by the medical segmentation decathlon (http://medicaldecathlon.com/) <br />
You can directly download the full body cts and segmentation maps from: <br />
https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing


## Oversampling to tackle strong class imbalance
Lung tumors are often very small, thus we need to make sure that our model does not learn a trivial solution which simply outputs 0 for all voxels.<br />
In this notebook we will use oversampling to sample slices which contain a tumor more often.

To do so we can use the **WeightedRandomSampler** provided by pytorch which needs a weight for each sample in the dataset.
Typically you have one weight for each class, which means that we need to calculate two weights, one for slices without tumors and one for slices with a tumor and create list that assigns each sample from the dataset the corresponding weight

## UNET
The idea behind a UNET is that we have "Downconvolutions" which are reducing the size of the image combined with increasing filter size followed by "Upconvolutions" which increase the image size up to the original size while reducing the number of filters. <br />
All pairs between Up- and Downconvolutions are linked with skip connections.<br />
Upsampling can either be done by interpolation or by UpConvolutions (ConvTranspose2d)


## GPU: NVIDIA GeForce GTX 1050 Ti

If you want to see complete result with video, you can use HTML files

# PScL-2LSAESM
The code is the implementation of our method described in the paper “ Matee Ullah, Fazal Hadi, Jiangning Song, Dong-Jun Yu, PScL-2LSAESM: bioimage-based prediction of protein subcellular localization by integrating heterogenous features with the two-level SAE-SM and mean ensemble method”.
## (I) 1_FeatureExtractionCode
### (1)	data
There are two datasets:
#### (i)	Train dataset
The benchmark training dataset contains a total of 2,708 immunohistochemistry (IHC) images of seven different protein subcellular locations selected from the version 21 human protein atlas (HPA) database.
#### (ii)	Independent dataset
The independent dataset contains a total of 227 IHC images of seven different proteins selected from HPA. <br />
Please download the datasets from "https://drive.google.com/drive/folders/1wVYxV1ktB8N4QsHaS8C_zTJfILtoPhG1?usp=sharing" and copy it to "data" folder.
### (2)	lib
lib folder contains all the features extraction related necessary codes used in this study.<br />
## (II)	2_FeatureSelectionCode
2_FeatureSelectionCode folder includes the following files
### (1)	SDAFeatureSelect
SDAFeatureSelect folder contains all required files related to Stepwise Discriminant Analysis.
### (2) SDA_FeatSelect.m
SDA_FeatSelect.m is the matlab file which calls the SDA feature selection algorithm.
## (III)	3_ClassificationCode
3_ClassificationCode folder includes all the libraries for SAE-SM classifier and 2L-SAE-SM framework.
## (IV)	Biomage_Feature_Extraction.m
Biomage_Feature_Extraction.m is the matlab file for extracting <br />
(1) Subcellular location features (Slfs) which includes
	(i)		DNA distribution features <br />
	(ii)	Haralick texture features <br />
(2)	Local binary pattern <br />
(3)	Completed local binary patterns <br />
(4)	Rotation invariant co-occurrence of adjacent LBP <br />
(5)	Locally encoded transform feature histogram and <br />
## (V)	twoL_SAE_SM.m
twoL_SAE_SM.m is the MATLAB file for the implementation of 2L-SAE-SM Framework.
## (VI)	Contact
If you are interested in our work or if you have any suggestions and questions about our research work, please contact us. E-mail: khan_bcs2010@hotmail.com.
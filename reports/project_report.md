----------------------------------------------------------------------------------
# Discussion Report

### Overview
K-Nearest Neighbors (KNN) is a classification technique used to predict the class of a data point based on the classes of the “K” number of data points in the training set closest to the data point of interest: the K-nearest neighbors (KNNs) (GRAD504, 2026). For categorical data, class prediction is a simple majority vote of the KNNs (GRAD504, 2026). Regression can be performed on numeric data such that the predicted value is the mean of the KNNs (GRAD504, 2026).

The defined number of neighbors (K) used in prediction determines the decision boundary of each prediction. As shown in Figure 1, lower K values complicate the decision boundary, and models with lower K values are more sensitive to noise such that erroneous data may yield false predictions for new data points (GRAD504, 2026). However, while higher K values simplify the decision boundary and reduce noise sensitivity, there can be an impact on accuracy as the overall majority class of the training set is predicted for most new data points (GRAD504, 2026). In defining K, Occam’s Razor should be applied to capture sufficient complexity while using the simplest model possible (GRAD504, 2026).

A variety of distance metrics can be deployed to determine the KNNs of a given input data point. The equation below shows the Euclidean distance formula, which is one of the most common distance metrics used for continuous data in KNN (GRAD504, 2026).

### KNN Implementation
#### Binary Class Prediction
Before developing the KNN, the dataset was cleaned to ensure class values were of the allowed classes (validateXY). For the initial implementation using the data included in Prog1data.csv, the allowed classes were either “+” or “-”. Cleaning involved the removal of one instance with an invalid class label. Of the remaining instances, 49.975% were of the “+” class, and 50.025% were of the “-” class (count_classes).

The KNN class (KNN_euclidean) was developed with 3 critical functions for model training (train), predicting classes of individual data points (predict_class), and for predicting classes of multiple new data points (predict). Training the KNN classifier involves storing all data points from the training set. For the initial implementation using the provided Prog1data.csv, the data was divided such that 80% (1599 instances) was used for training with the remaining 20% (400 instances) for testing. The instances were randomized before dividing into the training and test sets (train_test_splits). As shown in Table 1, both the training and test sets were representative of the overall representation of each class in the dataset.

*Table 1: Binary KNN Classifier Dataset Class Splits*

| Class | Overall Dataset | Training Set | Test Set |
|:------|----------------:|-------------:|---------:|
| +     | 49.975%         | 50.031%      | 49.750%  |
| -     | 50.025%         | 49.969%      | 50.250%  |

For predicting, the Euclidean distance function (euclidean_distance) was implemented as the distance metric. Prediction requires calculating the distance between every point of the training set and an input data point to determine the KNNs. Error handling was implemented so that if the chosen K value exceeded the number of items in the training set, all training instances were used for prediction. Evidently, computation happens during prediction, resulting in longer processing times for each prediction compared to some other classification techniques, like neural networks where the bulk of computation occurs during model training (GRAD504, 2026). Based on the implementation using the provided Prog1data.csv, the KNN classifier with Euclidean distance takes about 0.8 seconds to make a prediction for a single new data point and about 4.5 seconds to make a prediction on all 400 instances in the test set. The training dataset used is relatively small, yet the prediction time is still considerable. In real-world applications, training sets are often much larger, which would further increase computation time during prediction. In applications like autonomous driving and real-time cybersecurity threat mitigation where prediction speed is critical to safety, KNNs are likely not an appropriate machine learning technique.

The test set accuracy, precision, recall, and the F1 score of the binary KNN classifier are shown in Table 2. With the even split between “+” and “-” classes, accuracy is an appropriate performance metric and results suggests KNN is a good fit for this dataset (GRAD504, 2026). This is also reflected in the confusion matrix for the test set with K=5 (Figure 2). Precision indicates that when a given class is predicted, it is the correct class 98.477% of the time when K=5, and recall indicates out of all the instances of a given class, 97.487% were classified correctly when K=5. The high F1 score reflects both the high precision and high recall. As K increases, all evaluation metrics indicate a poorer model fit, with recall showing the greatest decline. However, the decline in performance metrics is subtle, with no change between K=15 and K=25, suggesting that a higher K value could be used for this dataset to reduce computational cost with minimal impact on performance.

*Table 2: Binary KNN Classifier Evaluation Metrics*

| Metric   | K=5    | K=15   | K=25   | K=35   |
|:---------|-------:|-------:|-------:|-------:|
| Accuracy | 0.98   | 0.9725 | 0.9725 | 0.9675 |
| Precision| 0.98477| 0.98454| 0.98454| 0.98438|
| Recall   | 0.97487| 0.9598 | 0.9598 | 0.94975|
| F1       | 0.9798 | 0.97201| 0.97201| 0.96675|

#### Multiclass Prediction
The Iris dataset (Iris.csv) was used to assess KNN performance with more than two class labels. The functions described for the binary KNN classifier were designed to support multiclass classification in addition to binary. Again, the dataset was cleaned to remove instances with invalid class labels (validateXY). The valid classes were defined as “iris-setosa,” “iris-versicolor,” and “iris-virginica.” All instances had valid classes in the Iris dataset. Class analysis (count_classes) indicated an even split with 33.333% of instances represented by each class.

A new KNN classifier was defined using the same Euclidean KNN class (KNN_euclidean) as described previously. The dataset was again randomly shuffled and divided to use 80% (120 instances) as the training set and 20% (30 instances) as test set (train_test_splits). As shown in Table 3, the training and test sets were representative of the overall representation of each class in the dataset.

*Table 3: Multiclass KNN Classifier Dataset Class Splits*

| Class            | Overall Dataset | Training Set | Test Set |
|:-----------------|----------------:|-------------:|---------:|
| Iris-setosa      | 33.333%         | 32.500%      | 36.667%  |
| Iris-versicolor  | 33.333%         | 34.167%      | 30.000%  |
| Iris-virginica   | 33.333%         | 33.333%      | 33.333%  |

For prediction, Euclidean distance was used to identify the KNNs with the same error handling for if the K value exceeded the number of items in the training set. Given the balanced class distribution, accuracy is an appropriate evaluation metric (GRAD504, 2026). If the dataset had been imbalanced, precision and recall would have been preferable as accuracy may otherwise have provided a misleading assessment of performance (GRAD504, 2026). Figure 3 shows accuracy plotted against K value. As shown, accuracy remains fairly stable for values of K between 5 and 75, suggesting that K could be increased as high as 75 to reduce computational cost without substantially affecting accuracy. However, when K is increased above 75, accuracy for both the training and test sets drops dramatically to around 30% when K=95. Increasing K too high causes the prediction to depend more on the overall class distribution of the training set than the local neighborhood of each point. Because “iris-versicolor” represents the greatest percent of the training set, as K increases, the model likely predicts that class for most predictions. This reduces accuracy toward approximately one third, which is the expected accuracy of always predicting a single class in a balanced three class dataset.

### Discussion
The discussed implementation used a brute-force approach to identify the KNNs, which involves calculating the distance between all pairs of points in the dataset. As the number of samples (N) and dimensions (D) of the dataset grows, computation scales by DN^2, such that this approach quickly becomes infeasible for large, high-dimensional datasets (scikit-learn, 2026).

The K-D tree data structure offers faster indexing for more efficient searches on large, high-dimensional datasets (scikit-learn, 2026). The required number of distance calculations is reduced by encoding aggregate distance information for the sample (scikit-learn, 2026). K-D trees use a binary tree structure that recursively partitions data along cartesian axes to divide the data into orthotropic regions (scikit-learn, 2026). The partitioning process is quick as the data is only divided along data axes rather than using higher dimensional distances (scikit-learn, 2026). The use of the median value for splitting regions as well as shrinking the data space to the actual data range offer further efficiencies in search but require more complicated model construction (SciPy, 2008). With K-D trees, computation scales by DNlog(N) (where D = dimensions and N = number of samples), which is a significant reduction compared to brute-force searches (scikit-learn, 2026). For datasets of less than 20-dimensions, K-D trees offer significant advantages over brute-force search, but as dimensions grow, K-D trees experience similar inefficiencies to brute-force (SciPy, 2008). The dataset used for the binary KNN classifier involved 11-dimenions, and the Iris dataset used for the multiclass KNN classifier involved only 4-dimensions. The use of K-D trees would therefore offer computational efficiency compared to the brute-force search that was used for the implementation described in this report.

Ball trees address the inefficiencies of K-D trees for higher-dimensional data (scikit-learn, 2026). While K-D trees partition along cartesian axes, ball trees divide the data in a series of nested hyper-spheres, offering more efficient searches on structured, high-dimensional data (scikit-learn, 2026). Tree construction is more computationally costly than K-D trees but significantly reduces the number of candidate points for each search (scikit-learn, 2026). A single distance calculation between the test point and centroid of the ball tree determines the lower and upper boundary on the distance from the test point to all points in the node (scikit-learn, 2026). For both the binary and multiclass KNN classifiers, the datasets were low-dimensional enough that a K-D tree should be efficient, making ball trees unnecessary for the applications discussed in this report.

On a typical workstation of 5GHz CPU, 128GB RAM, computation time is reasonable for the relatively small, low-dimensional datasets used to train the binary and multiclass KNN classifiers (~0.8 seconds per prediction for the binary KNN classifier). However, applications that require near real-time predictions would likely demand greater computational resources and more efficient data structures. Whether using brute-force or more advanced tree structures for KNN identification, computational cost increases substantially with dataset size and dimensionality, making deployment on a typical workstation impractical for large-scale or highly dimensional problems.

### References
Bzdok, D., Krzywinski, M., & Altman, N. (2018). Machine learning: supervised methods SVM and kNN. Nature Methods, 15(1), 5–6. https://doi.org/10.1038/nmeth.4551 

Clifton, C. (2026a). 1.08 - AI as functions and signal processing. Purdue University. Purdue University. Retrieved March 11, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344436/View. 

Clifton, C. (2026b). 3.04 - AI as ML: K-nearest neighbor. Purdue University. Purdue University. Retrieved March 23, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344456/View. 

Clifton, C. (2026c). 3.05 - K-nearest neighbor: K value. Purdue University. Purdue University. Retrieved March 23, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344479/View. 

Clifton, C. (2026d). 3.06 - K-nearest neighbor: implementation. Purdue University. Purdue University. Retrieved March 23, 2026, from https://purdue.brightspace.com/d2l/le/content/1493044/viewContent/21344480/View. 

Faiss. (n.d.). Welcome to FAISS documentation. Welcome to Faiss Documentation - Faiss documentation. https://faiss.ai/index.html 

scikit-learn. (2026a). 1.6. nearest neighbors. scikit learn. https://scikit-learn.org/stable/modules/neighbors.html 

scikit-learn. (2026b). BallTree. scikit learn. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree 

SciPy. (2008). KDTree. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html 

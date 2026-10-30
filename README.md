# Rice-Drought-Heat-Compact-Predictive-Gene-Panels

## A Feature Selection-Driven Pipeline to Identify Compact Predictive Gene Panels Characterizing Drought and Heat Stress Responses in Rice

This repository contains the source code used in the study:

**A Feature Selection-Driven Pipeline to Identify Compact Predictive Gene Panels Characterizing Drought and Heat Stress Responses in Rice**

### Overview

Rice (*Oryza sativa* L.) is one of the world's most important staple crops, yet its productivity is increasingly threatened by drought and heat stress. This repository provides the complete computational workflow used to identify compact predictive gene panels associated with drought and heat stress responses in rice.

The framework integrates RNA-seq data, feature selection methods, and classifiers to discover biologically meaningful and highly predictive gene signatures.

The final predictive gene panels consisted of:

* **10 genes** for drought stress classification
* **4 genes** for heat stress classification

Among all evaluated approaches, the combination of **Mutual Information-Minimum-Redundancy Maximum-Relevance (MImRMR)** feature selection and **Multi-Layer Perceptron Neural Network (MLPNN)** achieved the highest predictive performance.

---

## Workflow



![image alt](https://github.com/PlantSystemsBio/Rice-Drought-Heat-Compact-Predictive-Gene-Panels/blob/ee38e52c09135785bae66c9880ebac051a759a7b/Fig.%202.jpg)


---

## Feature Selection Methods

The repository includes implementations of the following feature selection approaches:

### 1. Stability Selection (SS)

A subsampling-based feature selection framework that combines repeated random sampling with sparse logistic regression to identify features consistently selected across data partitions.

### 2. Boruta

An all-relevant feature selection method based on Random Forests and shadow features, designed to identify all biologically relevant genes associated with stress responses.

### 3. Recursive Feature Elimination with Cross-Validation (RFECV)

Wrapper-based methods that iteratively remove less informative features and identify optimal gene subsets through cross-validation.

### 4. Mutual Information (MI)- Minimum-Redundancy Maximum-Relevance (MImRMR)

An information-theoretic method that ranks genes according to their statistical dependence with the stress phenotype. Then simultaneously maximizes relevance to the target phenotype while minimizing redundancy among selected genes.

### 5. Embedded Gradient Boosting Feature Selection (EGBFS)

An embedded approach that ranks genes according to feature importance scores derived from gradient boosting classifiers.

### 6. Genetic Algorithm (GA)

An evolutionary optimization approach that searches for optimal gene subsets using selection, crossover, and mutation operators.

---

## Classifiers

### 1. Multi-Layer Perceptron Neural Network (MLPNN)

Fully connected neural networks with extensive hyperparameter optimization including:

* Hidden layer architecture
* Learning rate
* Activation function

### 2. Support Vector Machine (SVM)

classifying observations by identifying a decision boundary that maximizes the separation between classes.

---

## Repository Structure

```text
rice-stress-predictive-gene-panels/
│
├── README.md
├── LICENSE
├── requirements
│
├── feature_selection.py
│   ├── SS
│   ├── Boruta
│   ├── RFECV
│   ├── MImRMR
│   ├── EGBFS
│   └── GA
│
├── deep_learning_classifiers.py
    ├── MLPNN
    ├── SVM
    
```

---

## Software Requirements

The analyses were performed using Python and the following libraries:

```text
numpy
pandas
scikit-learn
tensorflow
keras
matplotlib
seaborn
xgboost
boruta
joblib
scipy
```

## Reproducibility

- Exposure-time-group-based data splitting and cross-validation.
- Training-only preprocessing, feature selection, and model optimization.
- Application of preprocessing parameters derived from training data to held-out test and independent validation datasets.
- Comprehensive evaluation using held-out test and independent validation datasets.
- Standardized performance assessment using TP, TN, FP, FN, accuracy, precision, recall, and F1-score.

## Disclaimer

This repository contains computationally intensive feature selection and deep learning pipelines designed for research-scale RNA-seq analyses. Some workflows involve extensive hyperparameter searches, repeated cross-validation, evolutionary optimization, and neural network training.

Users are strongly encouraged to execute these analyses on high-performance computing (HPC) systems, dedicated workstations, or cloud computing platforms. Running the full workflow on resource-limited personal computers may result in excessive execution times, high memory consumption, or process termination due to insufficient computational resources.

## Author

**Dr. Masoud Shahsavari**

Department of Agronomy and Plant Breeding
College of Agriculture and Natural Resources
University of Tehran
Karaj, Iran

Email: [mshahsavari@ut.ac.ir](mailto:mshahsavari@ut.ac.ir)

---

## License

This project is released under the MIT License.

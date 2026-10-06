Federated TinyML on Multi-Layer Edge Architecture for Real-Time Cardiovascular Prediction
Authors
Yahya Dorostkar Navaei, Yaser Ahangari Nanehkaran
Description
This repository presents a Python-based implementation of an intelligent cardiovascular prediction framework inspired by Federated TinyML and multi-layer edge computing architectures. The study focuses on lightweight yet accurate predictive modeling suitable for real-time deployment at the edge, while maintaining robustness through ensemble learning strategies. Although the experiments are executed in a centralized simulation environment, the design principles and evaluation protocol reflect federated and edge-oriented constraints such as latency, scalability, and limited computational resources.
The proposed framework integrates shallow neural networks with classical machine learning models and ensemble strategies, enabling efficient cardiovascular risk prediction under realistic deployment assumptions.
Dataset Information
The experiments utilize the UCI Heart Disease dataset, a well-established benchmark in medical decision support systems.
•	Repository: UCI Machine Learning Repository
•	The datasets analyzed for this study can be found in the Kaggle repository. Heart Disease Dataset, available at:
https://www.kaggle.com/datasets/sukhmandeepsinghbrar/heart-attack-dataset
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
https://www.kaggle.com/datasets/prokashbarmancu/iomt-alert
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset.

The datasets contain clinical and demographic attributes related to cardiovascular conditions. All numerical attributes are standardized prior to model training. The target variable represents the presence or absence of heart disease.
Code Information
The codebase is organized as a single executable Python script that implements the full experimental pipeline, including data preprocessing, model training, evaluation, and visualization.
Implemented Models
•	Multi-Layer Perceptron (MLP)
•	Bagging-based MLP ensemble
•	Random Forest + MLP soft voting ensemble
•	XGBoost + MLP soft voting ensemble
•	Stacking ensemble using Logistic Regression as meta-learner
Each model is evaluated under repeated experimental runs to ensure statistical stability.
Methodology
Data Preprocessing
•	Removal of non-numeric attributes
•	Feature standardization using z-score normalization
•	Stratified train–test split (70% training, 30% testing)
Model Training
A compact MLP architecture is employed to emulate TinyML constraints. Ensemble learning techniques, including bagging, soft voting, and stacking, are used to improve predictive robustness without significantly increasing model complexity.
Experimental Design
The experiments are repeated for K = 20 independent runs with fixed random seeds. For each run, performance is evaluated on progressively increasing portions of the test set to simulate streaming or edge-level inference scenarios.
Assessment Metrics
The following metrics are reported and justified due to their relevance in medical and edge-based prediction tasks: - Accuracy: Measures overall classification correctness. - Precision: Indicates the reliability of positive predictions, critical for reducing false alarms. - Recall (Sensitivity): Reflects the ability to correctly identify patients with cardiovascular disease. - F1-score: Provides a balanced measure between precision and recall. - AUC (Area Under the ROC Curve): Evaluates discriminative power independent of decision thresholds. - Latency: Captures inference time, a key constraint in real-time and edge computing environments.
All metrics are computed during the Materials & Methods phase and aggregated across runs.
Usage Instructions
1.	Install Python 3.11.9.
2.	Clone this repository.
3.	Place heart.csv in the project root directory.
4.	Install the required dependencies.
5.	Execute the main script:
 	python main.py
6.	Performance metrics and visualization plots will be generated automatically.
Requirements
•	Python 3.11.9
•	Operating System: Windows 10 (64-bit)
•	Hardware: ASUS N56J Laptop, Intel Core i7, 8 GB RAM
Required Python Libraries
•	numpy
•	pandas
•	matplotlib
•	scikit-learn
•	tensorflow
•	xgboost
Computing Infrastructure
All simulations were conducted on a laptop system with the following configuration: - Processor: Intel Core i7 - Memory: 8 GB RAM - Operating System: Windows 10 (64-bit) - Software Environment: Python 3.11.9
Reproducibility
To facilitate reproducibility: - Fixed random seeds are used for NumPy and TensorFlow. - All hyperparameters are explicitly defined in the script. - The dataset source and preprocessing steps are fully documented. - The complete experimental pipeline can be reproduced by running the provided code.
Conclusions
The results demonstrate that lightweight neural architectures combined with ensemble learning strategies can achieve competitive cardiovascular prediction performance while maintaining low inference latency. This supports the feasibility of deploying such models within federated and multi-layer edge computing environments.
Limitations
•	The study is validated on a single public dataset.
•	Federated learning is simulated conceptually rather than implemented with distributed clients.
•	Communication overhead and privacy-preserving mechanisms are not explicitly modeled.
•	Deeper neural architectures are avoided to preserve TinyML constraints.
Citations
If this work or dataset is used in academic research, please cite: - UCI Machine Learning Repository: Heart Disease Dataset - Relevant literature on TinyML, federated learning, and ensemble-based medical prediction systems.
License
This project is intended for academic and research use only. Redistribution or commercial use requires permission from the authors.
Contribution Guidelines
Contributions are welcome for extending the framework toward true federated implementations, additional datasets, or deployment-oriented optimization. All contributions should include clear documentation and experimental justification.

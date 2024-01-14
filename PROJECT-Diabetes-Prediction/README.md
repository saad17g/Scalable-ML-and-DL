# Diabetes prediction model
## Diagram of the Application
![image](https://github.com/saad17g/Scalable-ML-and-DL/assets/96499884/6023f4f2-ff36-4c9e-b976-39fe321384e8)
## UI Links

- Prediction App: https://huggingface.co/spaces/saad177/Diabetes-Prediction
- Monitoring App: https://huggingface.co/spaces/saad177/Diabetes-Prediction-Monitoring

The prediction UI contains 3 parts. 
- The input part, where the user fills  the values for the 4 features
- The prediction part, where we output the model's prediction
- The explainability part, where we plot different graphs to explain the model's decision, for eg. which feature was the most impactful for the given prediction.

## Dataset

[Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Diabetes Exploratory Data Analysis

During EDA, we encoded categorical values, scaled numeric values, calculated correlation and identified the following
key features that contribute to the diabetes prediction:

* blood_glucose_level
* HbA1c_level
* age
* bmi

More details can be found
here: [diabetes_eda_and_backfill_feature_group.ipynb](diabetes_eda_and_backfill_feature_group.ipynb)

The initial dataset was significantly unbalanced:

![img.png](images/img.png)

So, we decided to use [YData Synthetic](https://github.com/ydataai/ydata-synthetic) to oversample the minority class.

We trained a CTGAN model and saved it to Hopsworks. We also generated new samples, which gave us a more balanced
dataset:

![img_1.png](images/img_1.png)

Besides, we added **data validation** by identifying a min-max range for our key features.

More details can be found
here: [v2_diabetes_eda_and_backfill_feature_group.ipynb](v2_diabetes_eda_and_backfill_feature_group.ipynb)

## Training

We testes KNeighborsClassifier, SGDClassifier, BernoulliNB, RandomForestClassifier and SVC. `RandomForestClassifier`
showed the best accuracy.

The overall accuracy on the imbalanced dataset was high. However, `recall` for diabetes prediction was relatively low.

![img_2.png](images/img_2.png)

With oversampling, we managed to improve the modal prediction for both classes:

![img_3.png](images/img_3.png)

Both models are saved to Hopsworks.

More details can be found
here: [diabetes_training_pipeline.ipynb](diabetes_training_pipeline.ipynb)

## Daily pipeline

We used YData model, that was initially trained for oversampling, to generate new daily records. The job is run on Modal
and saves 1000 records to Hopsworks feature store every day.

More details can be found
here: [diabetes-feature-pipeline-daily.py](diabetes-feature-pipeline-daily.py)

## Users data usage:

If consented, we collect users data and save it in a feature group on Hopsworks. This data is used for 2 purposes:

- Monitoring resources (eg. Confusion matrix) used for the monitoring app
- Retraining the model, on-demand through the same training pipeline, by uncommenting a part in the code. In this case, we retrain the model on the initial dataset + users data

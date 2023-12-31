# Scalable-ML LAB1:

### URL for gradio wine application:

**https://huggingface.co/spaces/saad177/wine**

### URL for gradio monitoring application:

**https://huggingface.co/spaces/saad177/wine-monitoring**

The objective of this lab is to deploy a serverless ML prediction system composed by many applications, using technologies such as HuggingFace, Hopsworks and Modal.
The system is composed of several pipelines, each responsable for a specific task:

- **Feature Pipelines** :

  - wine-eda-and-backfillfeature-group.ipynb
    - Responsible for reading the initial dataset, formatting it to remove null values/outliers (using the mean of the specific column), then storing it as a feature group on Hopsworks
  - wine-feature-pipeline-daily.py
    - Responsible for generating one new synthetic wine everyday, in the same format as the stored dataset on Hopsworks, then append it on the feature group.

- **Training Pipeline**:

  - wine-training-pipeline.ipynb
    - Responsiblee for training a model and testing it using the training/test split, then saving the model on Hopsworks along with a performance metrics (RMSE fo eg.)

- **Inference Pipeline**:

  - wine-batch-inference-pipeline.py
    - Responsible for infering the new data that was stored on Hopsworks (via the daily pipeline). This runs everyday and predict the added data.

- **UI and Monitoring**:

  - huggingface spaces appplications
    - The UI allows the user to insert some characteristics of a wine, then predicts the quality of this wine
    - The Monitoring app displays the prediction of the last added wine, a history of the 4 last predicted wines, as well as a confusion matrix showing the overall quality of the model.

- **Dataset Preparation for the Wine Dataset**:

  - First, we filled all the empty values in our Dataframe
  - We then detected outliers using zscore, and dropped all the outliers (around 700 rows)
  - We encoded the "type" using LabelEncored to 0 and 1 for white and red
  - We printed the correlation of each column to the quality target column, we kept only the ones having a corrrelation > 0.2
  - This dropped all the columns except 4: Volatile Acidity, Chlorides, Density, Alcohol

- **Generating new synthetic wines**:

  - To generate a new wine, we did the following process:
    - We assumed that our 4 columns followed a normal distribution
    - For each training sample, we calculated the probability of each one of the 4 columns following the normal distributions.
      - For eg. for Sample 1, we take the column Alcohol, let's say equal to 10, we calculate the likelihood of 10 using the probability distribution of Alcohol
    - Then we have a weight (likelihood) for each possible value for each one of our 4 columns
    - When creating a new wine, we generate the values of each column using these weights.

- **Model Choice**:
  - We chose a Regression Model using Random Forest Regressor to predict the value of the quality. So in contrast with the original dataset, which only have round values for the quality (eg. 3, 4), our model predicts a float value. The RMSE of our model is ~0.7. When testing the model on several predictions, we observed that for the wines that have a quality of 5 or 6, our model was very accurate, for those with quality of 7 our model was overall accurate. And for the generated wines with a quality of 4 or 8, our model was not very accurate. This can be explained by the very small number of samples in the training dataset with quality of 4 and 8.

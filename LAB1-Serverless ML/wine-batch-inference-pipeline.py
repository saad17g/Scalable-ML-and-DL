import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("wine_daily_inference")
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image"]
    )

    @stub.function(
        image=hopsworks_image,
        schedule=modal.Period(days=1),
        secret=modal.Secret.from_name("wine"),
    )
    def f():
        g()


def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    import dataframe_image as dfi
    from PIL import Image
    from datetime import datetime
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    print("Retrieving the model")

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    print(batch_data)

    y_pred = model.predict(batch_data)
    offset = 1
    wine_quality = y_pred[y_pred.size - offset]
    print("Wine predicted: " + str(wine_quality))

    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read()
    label = df.iloc[-offset]["quality"]
    print("Wine actual: " + str(label))

    print("Retrieving the monitor group")
    monitor_fg = fs.get_or_create_feature_group(
        name="wine_predictions",
        version=1,
        primary_key=["datetime"],
        description="Wine Prediction/Outcome Monitoring",
    )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        "prediction": [wine_quality],
        "label": [label],
        "datetime": [now],
    }
    monitor_df = pd.DataFrame(data)
    print(monitor_df)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    print("Retrieving the history DF")
    history_df = monitor_fg.read(read_options={"use_hive": True})
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App

    print(history_df)
    history_df = pd.concat([history_df, monitor_df])

    # save the last 4 predicted values
    df_recent = history_df.tail(4)
    dfi.export(df_recent, "./df_recent.png", table_conversion="matplotlib")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    # save the last predicted value
    df_last = history_df.tail(1)
    dfi.export(df_last, "./df_last.png", table_conversion="matplotlib")
    dataset_api.upload("./df_last.png", "Resources/images", overwrite=True)

    predictions = history_df["prediction"]
    labels = history_df[["label"]]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 6 quality types
    rounded_predictions = pd.DataFrame({"rounded_prediction": predictions.round()})

    print("labels", labels)
    print("int pred", rounded_predictions)
    print(
        "Number of different wine predictions to date: "
        + str(rounded_predictions.value_counts().count())
    )
    print("Number of different wine labels: " + str(labels.value_counts().count()))
    if labels.value_counts().count() >= 5:
        results = confusion_matrix(labels, rounded_predictions)

        df_cm = pd.DataFrame(
            results,
            ["True 4", "True 5", "True 6", "True 7", "True 8"],
            ["Pred 4", "Pred 5", "Pred 6", "Pred 7", "Pred 8"],
        )

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print(
            "You need 6 different wine quality predictions to create the confusion matrix."
        )
        print(
            "Run the batch inference pipeline more times until you get 6 different wine predictions"
        )


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("wine_daily_inference")
        with stub.run():
            f()

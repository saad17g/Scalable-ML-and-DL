import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub("diabetes_daily_inference")
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image"]
    )

    @stub.function(
        image=hopsworks_image,
        schedule=modal.Period(days=1),
        secret=modal.Secret.from_name("diabetes"),
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

    project = hopsworks.login(project="SonyaStern_Lab1")
    fs = project.get_feature_store()
    # print("Retrieving the model")

    # mr = project.get_model_registry()
    # model = mr.get_model("diabetes_gan_model", version=1)
    # model_dir = model.download()
    # model = joblib.load(model_dir + "/diabetes_gan_model.pkl")

    ### Generate confusion matrix out of users data
    ## get users submitted data
    feature_group = fs.get_feature_group(name="diabetes_user_data", version=1)
    query = feature_group.select_all()
    user_data_df = pd.DataFrame(feature_group.read())

    ## users can submit data with "I don't know" for the diabetes feature, here we remove the rows with this value

    user_data_df = user_data_df[user_data_df["diabetes"] != "Don't know"]
    user_data_df["diabetes"] = user_data_df["diabetes"].replace({"yes": 1, "no": 0})

    # print(user_data_df)

    actual_labels = user_data_df["diabetes"]
    predicted_labels = user_data_df["model_prediction"]
    # print(actual_labels)
    # print(predicted_labels)

    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

    df_cm = pd.DataFrame(
        conf_matrix,
        ["True", "False"],
        ["Pred True", "Pred False"],
    )

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix_diabetes_gan.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload(
        "./confusion_matrix_diabetes_gan.png", "Resources/images", overwrite=True
    )

    # save the last 4 predicted values
    df_recent = user_data_df.tail(4)
    dfi.export(df_recent, "./df_diabetes_recent.png", table_conversion="matplotlib")
    dataset_api.upload("./df_diabetes_recent.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("wine_daily_inference")
        with stub.run():
            f()

import gradio as gr
import hopsworks
import joblib
import pandas as pd
import requests
from PIL import Image

project = hopsworks.login()
fs = project.get_feature_store()

print("trying to dl model")
mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(volatile_acidity, chlorides, density, alcohol):
    print("Calling wine function")
    df = pd.DataFrame(
        [[alcohol, chlorides, volatile_acidity, density]],
        columns=["alcohol", "chlorides", "volatile_acidity", "density"],
    )
    print("Predicting")
    print(df)
    res = model.predict(df)
    print(res)
    return res


demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with different values for these properties",
    allow_flagging="never",
    inputs=[
        gr.Number(label="Alcohol"),
        gr.Number(label="Chlorides"),
        gr.Number(label="Volatile Acidity"),
        gr.Number(label="Density"),
    ],
    outputs=gr.Number(label="Quality"),
)

demo.launch(debug=True)

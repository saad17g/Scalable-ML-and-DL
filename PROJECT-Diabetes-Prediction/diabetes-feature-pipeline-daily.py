import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("diabetes_daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "scipy", "ydata_synthetic"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"),
                   timeout=6000)
    def f():
        g()


def g():
    import hopsworks
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    project = hopsworks.login()
    fs = project.get_feature_store()
    diabetes_fg = fs.get_feature_group(name="diabetes_gan", version=1)

    print("Retrieving the model")
    mr = project.get_model_registry()
    model = mr.get_model("ydata_model", version=1)
    model_dir = model.download()
    model = RegularSynthesizer.load(model_dir + "/ydata_model.pkl")

    synth_data = model.sample(1000)
    print(synth_data.info())

    diabetes_fg.insert(synth_data)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("diabetes_daily")
        with stub.run():
            f()

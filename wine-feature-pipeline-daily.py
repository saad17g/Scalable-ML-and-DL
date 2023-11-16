import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub("wine_daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks", "scipy"])

    @stub.function(
        image=image,
        schedule=modal.Period(days=1),
        secret=modal.Secret.from_name("wine"),
    )
    def f():
        g()


def calculate_probabilities(dataframe):
    import scipy

    mu = dataframe["alcohol"].mean()
    sig = dataframe["alcohol"].std()
    dataframe["p_alcohol"] = dataframe["alcohol"].apply(
        lambda x: abs(scipy.stats.norm(mu, sig).pdf(x))
        if x > mu
        else abs(1 - scipy.stats.norm(mu, sig).pdf(x))
    )

    mu = dataframe["chlorides"].mean()
    sig = dataframe["chlorides"].std()
    dataframe["p_chlorides"] = dataframe["chlorides"].apply(
        lambda x: abs(scipy.stats.norm(mu, sig).pdf(x))
        if x > mu
        else abs(1 - scipy.stats.norm(mu, sig).pdf(x))
    )

    mu = dataframe["volatile_acidity"].mean()
    sig = dataframe["volatile_acidity"].std()
    dataframe["p_volatile_acidity"] = dataframe["volatile_acidity"].apply(
        lambda x: abs(scipy.stats.norm(mu, sig).pdf(x))
        if x > mu
        else abs(1 - scipy.stats.norm(mu, sig).pdf(x))
    )

    mu = dataframe["density"].mean()
    sig = dataframe["density"].std()
    dataframe["p_density"] = dataframe["density"].apply(
        lambda x: abs(scipy.stats.norm(mu, sig).pdf(x))
        if x > mu
        else abs(1 - scipy.stats.norm(mu, sig).pdf(x))
    )

    return dataframe


def get_random_wine(dataframe):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    random_quality = random.randint(3, 9)
    random_quality_str = str(random_quality)
    filtered = dataframe.query("quality==" + random_quality_str).reset_index(drop=True)

    wine_df = pd.DataFrame(
        {
            "alcohol": random.choices(
                filtered["alcohol"], weights=filtered["p_alcohol"]
            ),
            "chlorides": random.choices(
                filtered["chlorides"], weights=filtered["p_chlorides"]
            ),
            "volatile_acidity": random.choices(
                filtered["volatile_acidity"], weights=filtered["p_volatile_acidity"]
            ),
            "density": random.choices(
                filtered["density"], weights=filtered["p_density"]
            ),
        }
    )
    wine_df["quality"] = random_quality

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_df = pd.DataFrame(wine_fg.read())

    wine_df = calculate_probabilities(wine_df)

    wine_entry_df = get_random_wine(wine_df)

    wine_fg.insert(wine_entry_df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()

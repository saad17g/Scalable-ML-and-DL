# UI 

You can find the UI of this application in the following link: https://huggingface.co/spaces/saad177/whisper-ru

# Whisper

We used a model-centric approach to improve our model.
Initially, we updated the learning rate according to the [original paper](https://arxiv.org/pdf/2212.04356.pdf). However,
it turned out to be too high, and we lowered it according
to [these](https://github.com/vasistalodagala/whisper-finetune) recommendations.

The results can be compared here:

[Basic model](https://huggingface.co/SofiaK/training-v1/)

[Improved model](https://huggingface.co/SofiaK/training-v2)


We have also looked into a data-centric approach and found an [external dataset](https://huggingface.co/datasets/google/fleurs) library for different languages. However, since we already had a 35GB dataset, which resulted in 8 hours of training, we decided extending the dataset and training the model even longer was not feasible. Besides, we managed to improve the model just by tuning hyperparameters.

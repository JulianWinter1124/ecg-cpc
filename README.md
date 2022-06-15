# Classification of Electrocardiograms using Contrastive Predictive Coding

This is the code for my master thesis "**Classification of Electrocardiograms using Contrastive Predictive Coding**"
Contrastive Predictive Coding is an unsupervised learning approach from the paper [Representation Learning with Contrastive Predictive Coding
](https://doi.org/10.48550/arXiv.1807.03748) which learns data representation in latent space by using autoregressive models and is suitable for many data modalities.

# Abstract from my work:

The following work deals with classifying multilabel electro-cardiographic data
using a “self-supervised” learning method called Contrastive Predictive Coding
(short CPC) [[1]](https://doi.org/10.48550/arXiv.1807.03748), that learns lower dimensional data representations without requir-
ing class labels. These data representations can then be used in classification tasks.
Learning from data without labels is potentially very beneficial in fields where cor-
rectly labeled data is scarce and only obtainable slowly by human experts.

The goal of this work is to re-implement CPC, apply and test several changes to
the architecture and evaluate its capability of classifying ECG-data. We try to answer
the following central questions to shine a light on CPC’s strengths and weaknesses:
How does the CPC architecture compare against our baseline models in a fully
supervised setting? How capable is CPC’s pretraining? How good are the learned
representations on their own? Under what circumstances is CPC especially efficient
to use? Can the original architecture be improved? Does CPC learn visually verifiable
useful latent representations?

We will answer these questions by conducting various experiments that aim to
emulate real-life circumstances such as low label availability and limited training
times. CPC is compared to fully supervised models, where we make use of architectures like the TCN [[2]](https://doi.org/10.48550/arXiv.1803.01271) and a special time-series classification focused residual net [[3]](https://doi.org/10.48550/arXiv.1511.07122),
but also introduce a wide variety of own baseline models.

For evaluation we look at predictions both quantitatively, through different met-
rics, and qualitatively, by either inspecting embeddings in lower dimensions of the
learned representations or by visualizing the prediction probabilities in a wide va-
riety of plots. Additionally we utilize the gradient of selected trained networks to
show spots in the input data which possibly contain ECG-classes.

Our contributions to the field are a more precise explanation of the original publication [[1]](https://doi.org/10.48550/arXiv.1807.03748), alterations to the one dimensional architecture suited for timeseries classification, namely non overlapping data windows, normalized latent representations,
different encoder-, autoregressive context- and downstream-task networks. We also
introduce a "no-context" architecture which predict latents without making use of
the standard context recurrent network. In total we trained and tested over 1700
networks and will also release their model properties and average scores as table for
all those interested.

Last but not least our generated prediction plots and "point of interest"-
visualization mainly based on grad-cam [[4]](https://doi.org/10.48550/arXiv.1610.02391), show the different models’ strengths
beyond classification scores and could also assist cardiologists in the future.

# This Repository:
[architectures_cpc](architectures_cpc) includes all Pytorch models using Contrastive Predictive Coding

[architectures_baseline_challenge](architectures_baseline_challenge) includes all baseline Pytorch models

[architectures_baseline_challenge](architectures_various) includes Pytorch models that were used to generate to generate the Points of Interest in the input data

[external](external) other repositories (not my code) with only minor changes. (e.g TCN network)

[jupyter_notebooks](jupyter_notebooks) various Jupyter notebooks containing experiments and visualizations used throughout the thesis.

[models](models) (local) folder with trained models. Empty due to data limitations

[main_cpc_benchmark_test.py](main_cpc_benchmark_test.py) tests specified models

[main_cpc_benchmark_train.py](main_cpc_benchmark_train.py) trains specified models

[main_cpc_explain.py](main_cpc_explain.py) Points of Interest images generation for specified models

[main_cpc_explain_gradcam.py](main_cpc_explain_gradcam.py) Points of Interest images generation (more based on GradCam) for specified models

[main_produce_plots_for_tested_models.py](main_produce_plots_for_tested_models.py) Generates most plots and tables used in the thesis for specified models


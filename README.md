# Cohort Shapley
Cohort Shapley (Shapley cohort refinement) is a local explanation method for black box prediction models using Shapley value from cooperative game theory. Cohort Shapley naturally decomposes statistical ANOVA based global sensitivity analysis (variance explained Shapley or Shapley effect) in uncertainty quantification.

Cohort Shapley computes variable importance by including or excluding subjects similar to the target subject on that variable to form a similarity cohort and applying Shapley value to the cohort averages. Cohort Shapley is applicable to observational data without the model, or we only need single prediction runs for all data subjects. The variables for explanation can be different from the input variables of training time. That would fit to model auditing and understanding.

See details for the [paper](https://arxiv.org/pdf/1911.00467.pdf):
> Mase, M., Owen, A. B., & Seiler, B. (2019). Explaining black box decisions by Shapley cohort refinement. arXiv preprint arXiv:1911.00467.

# Install
Install the package locally with pip command.
```bash
git clone https://github.com/cohortshapley/cohortshapley
pip install -e cohortshapley
```

## Prerequisites
This code is tested on:
- Python 3.6.9
- pip 20.2.4
- NumPy 1.13.4
- Pandas 1.1.5
- scikit-learn 0.24.0
- matplotlib 3.3.3
- tqdm 4.54.1

For example notebooks, we need:
- jupyter 1.0.0
- XGBoost 1.0.0

Dockerfile for prerequisite software is available in [dockerfile](dockerfile) directory.

# Getting Started
See Jupyter notebook examples in [notebook](notebook) directory.

# Usages
## Variance Shapley for global explanation
This is an implementation of Shapley value for global explanation, also known as Shapley effect. Contribution of each variable is computed as variance explained.
It is easy to use with binning for numerical integration:
```python
similarity.bins = 10 # number of bins used for numerical integration
vs_values = vs.VarianceShapley(f(X.values),similarity.binning(X.values)[0])
```

## Cohort Shapley for local explanation
This is an implementation of cohort Shapley for local explanation.
This package computes Shapley values of impact and squared impact. Ordinary impact version indicates contribution of each input predictor variable to difference between predicted value and prediction average. Squared version represents contribution to squared difference between predicted value and prediction average, that splits variance Shapley value into an individual subject level.

### similarity function
Firstly, choose a similarity function for each input variable that is used in cohort refinement. Typical functions are prepared in [cohortshapley.similarity](cohortshapley/similarity.py) module.
- similar_in_distance(): the subjects are similar if the distance is within a predefined ratio (default: 0.1) of defined value range of the variable.
- similar_in_distance_cutoff():
the subjects are similar if the distance is within a predefined ratio (default: 0.1) of xth to (100-x)th percentile (x is a predefined cutoff value, default:0.1) of defined value range of the variable.
- similar_in_samebin():
the subjects are similar if they are in same bin after binning().

### compute from prediction result or model
Then, compute Shapley value from prediction result:
```python
similarity.ratio = 0.1
cs_obj = cs.CohortShapley(None, similarity.similar_in_distance_cutoff,
                          subject_id, subject, predY)
cs_obj.compute_cohort_shapley()
```
or from model:
```python
similarity.ratio = 0.1
cs_obj = cs.CohortShapley(f, similarity.similar_in_distance_cutoff,
                          subject_id, subject)
cs_obj.compute_cohort_shapley()
```
Then you get cohort Shapley value in cs_obj.shapley_value and squared cohort Shapley value in cs_obj.shapley_values2.

## Baseline Shapley
This is a naive implementation of conventional Shapley value for theoretical comparison. The baseline Shapley explains the difference between predicted value and prediction on baseline points. We can select multiple baseline points. If we select all the data points (all baseline Shapley), the baseline prediction is the prediction average.
```python
bs_obj = bs.BaselineShapley(f, subject, baseline)
bs_obj.compute_baseline_shapley()
```
Then you get baseline Shapley value  in bs_obj.shapley_value and squared baseline Shapley value in bs_obj.shapley_value2.

# Visualization
This package includes visualization of aggregated Shapley values for all subjects. See [notebook](notebook).

# Acknowledgements
The algorithms and visualizations used in the repository are developed in collaborative research of Stanford University and Hitachi, Ltd.

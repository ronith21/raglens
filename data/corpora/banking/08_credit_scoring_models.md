# Credit Scoring Models

A credit score estimates how likely an applicant is to repay. Banks use scores to approve or decline applications, set limits and price loans.

## Scorecards

The traditional retail scorecard is a logistic regression model. Continuous variables such as income or age are grouped into bins, and each bin is converted into its **Weight of Evidence (WoE)**, the log ratio of the share of good borrowers to the share of bad borrowers in that bin. The **Information Value (IV)** of a variable summarises how well its bins separate good from bad borrowers and is used to select variables. The final model is scaled into points, so that a loan officer can see how much each characteristic adds to or subtracts from the score.

## Measuring Discrimination

Model quality is usually measured with the area under the ROC curve (AUC) or the closely related **Gini coefficient**, where Gini = 2 × AUC - 1. A Gini of 0 means the model cannot separate good from bad borrowers; higher values mean better ranking. Calibration is checked separately: predicted default rates should match observed default rates across score bands.

## Monitoring

After deployment, the population applying for credit changes over time. The **Population Stability Index (PSI)** compares the distribution of scores or variables in recent applications with the development sample. A rule of thumb treats a PSI below 0.1 as stable, between 0.1 and 0.25 as a moderate shift worth investigating, and above 0.25 as a significant shift that may require redevelopment.

## Reject Inference

A model is trained only on applicants who were approved, because the bank never observes whether rejected applicants would have repaid. Reject inference techniques try to correct this selection bias, for example by assigning inferred outcomes to rejected applications.

## Machine Learning Models

Gradient boosting models often discriminate better than scorecards, but they are harder to explain. Banks that use them typically add explanation techniques such as SHAP values, so that each decision can be broken down into the contribution of each input. Explanations are needed to give applicants reasons for an adverse decision and to satisfy supervisors.

## Model Risk Management

Credit models are subject to model risk management: independent validation before use, documentation of data, assumptions and limitations, ongoing performance monitoring and periodic revalidation. Fairness is also reviewed, because a model must not discriminate on protected characteristics, directly or through proxies.

# This is the code used for our paper: Learning Discriminative Features for Adversarial Robustness
Our implementations for adversarial attacks, defenses, and discriminative loss functions can be found in this repository.

## Config.py
This file contains some hyperparameters used for training and testing our models. 
* train_mode: whether to train the model normally, using adversarial training (at), or with adversarial logit pairing (alp)
* test_mode: when the model predicts the validation data, this specifies what attack should be conducted (clean for no attack).
* test_bb: set to true for using surrogate model black-box attacks
* bb_metric: specify which surrogate model to use for the black-box attack
* metrics: this parameter is used when metric is set to "multiple" and specifies which models should be trained consecutively.
* metric: the discriminative loss function that optimizes the model.
* m: the m value for training aaml
* s: the s value for training aaml. For training multiple models with different s values, add multiple s values to the array.

## Training
Hyperparameters are specified in Config.py. Therefore, the model (or models) can be trained by running this command:

`python train.py`

## Testing
For running a single test, this can be run using the hyperparameters in Config.py

`python test.py`

For running multiple tests, this can be run using test_batch.py. Note, for convenience, hyperparameters in Config.py are overwritten. Any changes that would be made in Config.py should be made here.

`python test_batch.py`

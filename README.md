# misogyny-classifier

A binary classifier for misogynistic text.

## Dataset Preparation

1. The dataset provided contained a `nan` value in the `label` column which was removed.
2. The dataset contained slightly more of one category than the other. This was normalised so that both categories are of the same size.


## MisogynyModel - TF-IDF trained model

The basis of this model relies on creating a set of TF-IDF embeddings of the training set. The training of the model involves creating an average TF-IDF embedding of the misogynystic and non-misogynystic embeddings against an initial zero-vector for each.

The inference of the model requires generating a commensurate TF-IDF embedding with the training set, and then performing a cosine-similarity calculation against the misogynystic embedding and the non-misogynystic embedding. The greater of the two values determines the labels assigned to the input.

### Inference

Configuring this model for interence will require mapping any input string onto the TF-IDF space created during training. This will require dopping tokens that fall outside of the pre-defined vocabulary. However once done we can construct the relevant TF-IDF vector and compute the relevant values to classify the input string.

### Results

We use a k-fold repeated random subsampling (9:1) validation against the f1 score.

We find that the accuracy of the model is consistently between 60-70%.
The f1 score has a mean value ~0.65 and a standard deviation of ~0.02.

## Finetuned distilbert-base-cased

Using the HuggingFace library we can quickly compare our naive TF-IDF model with a BERT derived language model that is fine tuned for a binary classification task using our data set.

Using various factory methods we generate a tokenizer, and SequenceClassfication head version of the `distilbert-base-cased` model, optimizer (by default `Adam`)

We split the dataset into a (1:4) test/training set. We then train the model, as well as performing a hyperparameter optimisation.

### Inference

Configuring the model for inference is trivial, as we must simply load the pretrained model from disk, configure the relevant tokenizer, and simply call the relevant predict method for any tokenized input. The model will return a classification by definition.

### Results

The full `stdout` for the training can be found in the `bert_training_stdout.md` file.

We find the validation accuracy to be ~90%, significantly higher than our naive TF-IDF model.

Given the size of the data set, it is possible that the model has overfit, and a larger dataset would be needed to get a a better understanding about whether `distilbert-base-cased` is an appropriate model for this classification task.

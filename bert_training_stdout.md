# distil-bert-cased training stdout

```
Some weights of the model checkpoint at distilbert-base-cased were not used when initializing DistilBertForSequenceClassification: ['vocab_projector.weight', 'vocab_transform.weight', 'vocab_layer_norm.weight', 'vocab_transform.bias', 'vocab_projector.bias', 'vocab_layer_norm.bias']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-cased and are newly initialized: ['pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.96ba/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.84ba/s]
The following columns in the training set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
/home/james/working/misogyny-classifier/venv/lib/python3.10/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
***** Running training *****
  Num examples = 1654
  Num Epochs = 3
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 1
  Total optimization steps = 621
  Number of trainable parameters = 65783042
 33%|██████████████████████████████████████████████████████                                                                                                            | 207/621 [43:16<1:20:20, 11.64s/it]The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 414
  Batch size = 8
{'eval_loss': 0.2980162501335144, 'eval_accuracy': 0.8961352657004831, 'eval_runtime': 191.2829, 'eval_samples_per_second': 2.164, 'eval_steps_per_second': 0.272, 'epoch': 1.0}                           
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                      | 414/621 [1:29:43<40:04, 11.61s/it]The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 414
  Batch size = 8
{'eval_loss': 0.4074811637401581, 'eval_accuracy': 0.8743961352657005, 'eval_runtime': 191.5516, 'eval_samples_per_second': 2.161, 'eval_steps_per_second': 0.271, 'epoch': 2.0}                           
{'loss': 0.2548, 'learning_rate': 9.742351046698874e-06, 'epoch': 2.42}                                                                                                                                    
 81%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                               | 500/621 [1:51:15<25:18, 12.55s/it]Saving model checkpoint to test_trainer/checkpoint-500
Configuration saved in test_trainer/checkpoint-500/config.json
Model weights saved in test_trainer/checkpoint-500/pytorch_model.bin
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 621/621 [2:16:41<00:00, 11.59s/it]The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 414
  Batch size = 8
{'eval_loss': 0.4602499008178711, 'eval_accuracy': 0.9057971014492754, 'eval_runtime': 191.091, 'eval_samples_per_second': 2.167, 'eval_steps_per_second': 0.272, 'epoch': 3.0}                            
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 621/621 [2:19:52<00:00, 11.59s/it]
                                                                                                                                                                                                           
Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 8392.9385, 'train_samples_per_second': 0.591, 'train_steps_per_second': 0.074, 'train_loss': 0.21941625284879873, 'epoch': 3.0}                                                          
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 621/621 [2:19:52<00:00, 13.52s/it]
```
TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"
TRAINING_MAX_LENGTH = 512
TRAINING_STRIDE = 128
OUTPUT_DIR = "deberta_large"

# imports
import json
import argparse
from itertools import chain
from functools import partial

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader

from datasets import Dataset, features
import numpy as np
import os
from collections import Counter
import pandas as pd
from collections import defaultdict
from typing import Dict


data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/train.json"))

for i in range(len(data)):
    data[i]["fold"] = data[i]["document"]%4
    

# using fold%4=2 for validation set
train_data = []
val_data = []

for d in data:
    if d["fold"]==2:
        val_data.append(d)
    else:
        train_data.append(d)


# using external data such as mpware
add_mixtral_data = json.load(open("/kaggle/input/pii-mixtral8x7b-generated-essays/mpware_mixtral8x7b_v1.1.json"))
print("add mixtral data: ", len(add_mixtral_data))

for d in add_mixtral_data:
    if "I-USERNAME" not in d["labels"]:
        train_data.append(d)

# label-id mapping
all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)

# convert data into Dataset format
train_ds = Dataset.from_dict({
    "document": [str(x["document"]) for x in train_data],
    "tokens": [x["tokens"] for x in train_data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in train_data],
    "provided_labels": [x["labels"] for x in train_data],
})


val_ds = Dataset.from_dict({
    "document": [str(x["document"]) for x in val_data],
    "tokens": [x["tokens"] for x in val_data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in val_data],
    "provided_labels": [x["labels"] for x in val_data],
})


def get_labels(word_ids, word_labels):
    
    label_ids = []
    for word_idx in word_ids:                            
        if word_idx is None:
            label_ids.append(-100) # -100 is for CLS tokens etc.
        else:
            label_ids.append(label2id[word_labels[word_idx]])
    return label_ids

# to tokenize text with stride and overflow tokens
def tokenize(example, tokenizer, label2id, max_length=512, stride=128):
    
    encoded = tokenizer(example['tokens'],
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=stride,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True)

    encoded['labels'] = []

    encoded['wids'] = []
    n = len(encoded['overflow_to_sample_mapping'])
    for i in range(n):

        # Map back to original row
        text_idx = encoded['overflow_to_sample_mapping'][i]
        
        # Get word indexes (this is a global index that takes into consideration the chunking)
        word_ids = encoded.word_ids(i)
        
        # Get word labels of the full un-chunked text
        word_labels = example['provided_labels']
    
        # Get the labels associated with the word indexes
        label_ids = get_labels(word_ids, word_labels)
        encoded['labels'].append(label_ids)
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])
    
    res = []
    
    for i in range(len(encoded["input_ids"])):
        
        length = len(encoded.input_ids[i])
        labels = encoded["labels"][i]
        input_ids = encoded["input_ids"][i]
        token_type_ids = encoded["token_type_ids"][i]
        attention_mask = encoded["attention_mask"][i]
        wids = encoded["wids"][i]
        tokens = []
        provided_labels = []
        prev = -1
        for w in wids:
            if w !=-1:
                
                if prev==w:
                    continue
                
                prev = w
                tokens.append(example['tokens'][w])
                provided_labels.append(example['provided_labels'][w])
        
        res.append({"tokens": example['tokens'], "provided_labels": provided_labels,  "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "document": example["document"], "wids": wids, "labels": labels, "length": length})
    
    return res


# tokenize
train_result = []

for i in range(len(train_ds)):
    train_result.extend(tokenize(train_ds[i], tokenizer, label2id))

val_result = []

for i in range(len(val_ds)):
    val_result.extend(tokenize(val_ds[i], tokenizer, label2id))


# form final datatsets
train_res_ds = Dataset.from_dict({
	"tokens": [x["tokens"] for x in train_result],
	"input_ids": [x["input_ids"] for x in train_result],
	"provided_labels": [x["provided_labels"] for x in train_result],
	"token_type_ids": [x["token_type_ids"] for x in train_result],
	"attention_mask": [x["attention_mask"] for x in train_result],
	"document": [x["document"] for x in train_result],
	"wids": [x["wids"] for x in train_result],
	"labels": [x["labels"] for x in train_result],
	"length": [x["length"] for x in train_result],
})


val_res_ds = Dataset.from_dict({
	"tokens": [x["tokens"] for x in val_result],
	"input_ids": [x["input_ids"] for x in val_result],
	"provided_labels": [x["provided_labels"] for x in val_result],
	"token_type_ids": [x["token_type_ids"] for x in val_result],
	"attention_mask": [x["attention_mask"] for x in val_result],
	"document": [x["document"] for x in val_result],
	"wids": [x["wids"] for x in val_result],
	"labels": [x["labels"] for x in val_result],
	"length": [x["length"] for x in val_result],
})


# get reference df of validation set
df = pd.DataFrame(val_ds)
df['labels'] = df['provided_labels']
ref_df = df[['document', 'tokens', 'labels']].copy()
ref_df = ref_df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
ref_df['token'] = ref_df.groupby('document').cumcount()

reference_df = ref_df[ref_df['label'] != 'O'].copy()
reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
reference_df = reference_df[['row_id', 'document', 'token', 'label']].copy()


# Return model predictions after thresholding and in required output format
def parse_predictions(predictions, id2label, res_ds, threshold):
    
    pred_softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis = 2).reshape(predictions.shape[0],predictions.shape[1],1)
    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:,:,:12].argmax(-1)
    O_preds = pred_softmax[:,:,12]
    preds_final = np.where(O_preds < threshold, preds_without_O , preds)

    triplets = []
    row, document, token, label, token_str = [], [], [], [], []
    
    n = len(res_ds)
    for i in range(n):
        ds = res_ds[i]
        for j in range(0, len(ds['input_ids'])):

            if ds['wids'][j]!=-1:
                label_pred = id2label[preds_final[i][j]]
                token_id = ds['wids'][j]
                token_value = ds['tokens'][token_id]          

                if label_pred != "O":
                    triplet = (ds['document'], token_id, token_value)

                    if triplet not in triplets:
                        row.append(i)
                        document.append(ds['document'])
                        token.append(token_id)
                        label.append(label_pred)
                        token_str.append(token_value)
                        triplets.append(triplet)

    df = pd.DataFrame({
        "eval_row": row,
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })

    df = df.drop_duplicates().reset_index(drop=True)

    df["row_id"] = list(range(len(df)))
    return df

# Calculate metrics
class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


# Inspired from https://www.kaggle.com/code/thedrcat/pii-data-detection-train-with-w-b, https://www.kaggle.com/code/conjuring92/pii-metric-fine-grained-eval
def compute_metrics(p, id2label, valid_ds, valid_df, threshold=0.95):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    predictions, labels = p
    
    pred_df = parse_predictions(predictions, id2label, valid_ds, threshold=threshold)
    
    references = {(row.document, row.token, row.label) for row in valid_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}
    
    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1] # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:] # avoid B- and I- prefix
            
        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:] # avoid B- and I- prefix
        
        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()
    
    for prf in score_per_type.values():
        totals += prf

    results = {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "f1": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k!= 'O'},
    }
    
    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                if isinstance(v, dict):
                    for n2, v2 in v.items():
                        final_results[f"{key}_{n}_{n2}"] = v2
                else:
                    final_results[f"{key}_{n}"] = v              
        else:
            final_results[key] = value
            
    return final_results


# Training
model = AutoModelForTokenClassification.from_pretrained(
    TRAINING_MODEL_PATH,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=20,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    auto_find_batch_size=True
)

trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=train_res_ds,
    eval_dataset=val_res_ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, id2label=id2label, valid_ds=val_res_ds, valid_df=reference_df, threshold=0.95),#partial(compute_metrics, all_labels=all_labels),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Inference on validation set
preds = trainer.predict(val_res_ds)

# Compute metrics for different thresholds and choose threshold with best f5 metric
print('Computing final metrics...')
final_metrics = {
    f'final_f5_at_{threshold}': compute_metrics((preds.predictions, None), id2label, val_res_ds, reference_df, threshold=threshold)['f1']
    for threshold in [0.4, 0.5,0.6,0.7,0.8,0.9,0.95,0.97, 0.98, 0.99, 0.995, 0.998]
}
print(final_metrics)
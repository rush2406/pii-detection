model_path = '/kaggle/input/mixtral-pii/deberta_large'

# imports
import json
import argparse
from itertools import chain
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np

# load test data
data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/test.json"))

tokenizer = AutoTokenizer.from_pretrained(model_path)

# tokenize using overlapping tokens and stride
def tokenize(example, tokenizer, max_length=512, stride=128):
    
    with_labels = False
    
    encoded = tokenizer(example['tokens'],
                        is_split_into_words=True,
                        return_overflowing_tokens=True,
                        stride=stride,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True)

    encoded['wids'] = []
    n = len(encoded['overflow_to_sample_mapping'])
    for i in range(n):

        # Map back to original row
        text_idx = encoded['overflow_to_sample_mapping'][i]
        
        # Get word indexes (this is a global index that takes into consideration the chunking)
        word_ids = encoded.word_ids(i)
        
        encoded['wids'].append([w if w is not None else -1 for w in word_ids])
    
    res = []
    
    for i in range(len(encoded["input_ids"])):
        
        length = len(encoded.input_ids[i])
        input_ids = encoded["input_ids"][i]
        token_type_ids = encoded["token_type_ids"][i]
        attention_mask = encoded["attention_mask"][i]
        wids = encoded["wids"][i]
        
        # use all tokens because wids is with respect to it
        res.append({"tokens": example['tokens'], "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "document": example["document"], "wids": wids})
    
    return res

ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
})

tokenizer = AutoTokenizer.from_pretrained(model_path)
result = []

for i in range(len(ds)):    
    result.extend(tokenize(ds[i], tokenizer))

res_ds = Dataset.from_dict({
    "tokens": [x["tokens"] for x in result],
    "input_ids": [x["input_ids"] for x in result],
    "token_type_ids": [x["token_type_ids"] for x in result],
    "attention_mask": [x["attention_mask"] for x in result],
    "document": [x["document"] for x in result],
    "wids": [x["wids"] for x in result]
})

# label-id mapping
config = json.load(open(Path(model_path) / "config.json"))
id2label = config["id2label"]


from scipy.special import softmax

# model paths
paths = ["/kaggle/input/deberta-pii-foldwise/fold 2", "/kaggle/input/deberta-pii-extra/deberta_large", "/kaggle/input/pii-1850-mixtral-fold2-12/deberta_large", "/kaggle/input/pii-2kmistral/deberta_large"]
all_preds = []
t = [0.95, 0.99, 0.995, 0.95] # model threshold obtained from validation set

for p, threshold in zip(paths, t):

    model = AutoModelForTokenClassification.from_pretrained(p)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
    args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=1, 
        report_to="none",
    )
    trainer = Trainer(
        model=model, 
        args=args, 
        data_collator=collator, 
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(res_ds).predictions
    pred_softmax = softmax(predictions, axis = -1) 
    
    preds = pred_softmax.argmax(-1)
    preds_without_O = pred_softmax[:,:,:12].argmax(-1)
    O_preds = pred_softmax[:,:,12]
    
    local_preds_final = np.where(O_preds < threshold, preds_without_O , preds)
    all_preds.append(local_preds_final)

# in case of tie, use smallest class label: Falsely detecting PII is less harmful than not detecting PII at all according to f5 metric
def smallest_freq_elem(array):
    freq_count = np.bincount(array)
    idx_of_max_count = np.where(freq_count == np.max(freq_count))[0][0]
    return np.min(np.where(freq_count == freq_count[idx_of_max_count])[0])

# vote among all models
preds_final = np.apply_along_axis(lambda x: smallest_freq_elem(x), axis=0, arr=all_preds)

# get class labels
n = len(res_ds)
triplets = []
document, token, label, token_str = [], [], [], []

for i in range(n):
    ds = res_ds[i]
    for j in range(0, len(ds['input_ids'])):
        
        if ds['wids'][j]!=-1:
            label_pred = id2label[str(preds_final[i][j])]
            token_id = ds['wids'][j]
            token_value = ds['tokens'][token_id]          
            
            if label_pred != "O":
                triplet = (ds['document'], token_id, token_value)
                
                if triplet not in triplets:
                    document.append(ds['document'])
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(token_value)
                    triplets.append(triplet)


df = pd.DataFrame({
    "document": document,
    "token": token,
    "label": label,
    "token_str": token_str
})
df["row_id"] = list(range(len(df)))
display(df.head(100))

df[["row_id", "document", "token", "label"]].to_csv("submission.csv", index=False)
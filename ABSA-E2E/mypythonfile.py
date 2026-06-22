#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
import evaluate
from transformers import Trainer
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.metrics import classification_report


# In[2]:


import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("Running on CPU")


# In[3]:


# Load the XML data and convert it to a DataFrame
def xml_to_rows(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    rows = []

    for sentence in root.findall("sentence"):
        sid = sentence.get("id")
        text = sentence.find("text").text

        aspect_terms = sentence.find("aspectTerms")

        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                rows.append({
                    "id": sid,
                    "sentence": text,
                    "aspect": aspect.get("term"),
                    "polarity": aspect.get("polarity")
                })

    return pd.DataFrame(rows)


# In[4]:


myXML = xml_to_rows("./Restaurants_Train_v2.xml")


# In[5]:


myXML.shape


# In[6]:


myXML.head()


# In[7]:


csv_df = pd.read_csv("./Laptop_Train_v2.csv", encoding="utf-8")

csv_df = csv_df[[
    "id",
    "Sentence",
    "Aspect Term",
    "polarity"
]]

csv_df.columns = ["id", "sentence", "aspect", "polarity"]


# In[8]:


csv_df.head()


# In[9]:


csv_df.shape


# # Merge the datasets

# In[30]:


final_df = pd.concat([csv_df, myXML], ignore_index=True)


# In[31]:


final_df.head()


# In[32]:


final_df["sentence"].iloc[1]


# In[33]:


final_df["sentence"].iloc[1]


# In[34]:


final_df.shape


# In[35]:


print(final_df["polarity"].unique())


# # Preprocess

# In[36]:


final_df["polarity"] = final_df["polarity"].str.strip()
final_df["aspect"] = final_df["aspect"].str.replace('"', '')


# # Take 1000 of dataset

# In[37]:


final_df = final_df.head(6000)


# In[38]:


final_df.shape


# # BIO tagging 

# In[39]:


import pandas as pd
import string

# ---------------------------
# TOKENIZER (clean + simple)
# ---------------------------
def tokenize(text):
    return [
        t.strip(string.punctuation).lower()
        for t in text.split()
        if t.strip(string.punctuation)
    ]

# ---------------------------
# FIND ASPECT SPAN
# ---------------------------
def find_span(tokens, aspect_tokens):
    n, m = len(tokens), len(aspect_tokens)

    for i in range(n - m + 1):
        if tokens[i:i+m] == aspect_tokens:
            return i, i + m - 1

    return -1, -1

# ---------------------------
# BIOES TAGGING (PAPER STYLE)
# ---------------------------
def build_tags(tokens, start, end, polarity):
    tags = ["O"] * len(tokens)

    length = end - start + 1

    # SINGLE WORD → S
    if length == 1:
        tags[start] = f"S-{polarity}"

    # MULTI WORD → B I E
    else:
        tags[start] = f"B-{polarity}"

        for i in range(start + 1, end):
            tags[i] = f"I-{polarity}"

        tags[end] = f"E-{polarity}"

    return tags

# ---------------------------
# MAIN CONVERTER
# ---------------------------
def convert(df):
    dataset = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        aspect = row["aspect"]
        polarity = row["polarity"].upper()  # POSITIVE/NEGATIVE/NEUTRAL/CONFLICT

        tokens = tokenize(sentence)
        aspect_tokens = tokenize(aspect)

        start, end = find_span(tokens, aspect_tokens)

        tags = ["O"] * len(tokens)

        if start != -1:
            tags = build_tags(tokens, start, end, polarity)

        dataset.append(list(zip(tokens, tags)))

    return dataset


# In[40]:


# bio_data = convert(final_df)


# In[41]:


# bio_data[1]


# # Model training 

# In[42]:


import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report

# =========================================================
# 1. LABELS (BIOES + SENTIMENT)
# =========================================================
labels = [
    "O",
    "B-POSITIVE", "I-POSITIVE", "E-POSITIVE", "S-POSITIVE",
    "B-NEGATIVE", "I-NEGATIVE", "E-NEGATIVE", "S-NEGATIVE",
    "B-NEUTRAL", "I-NEUTRAL", "E-NEUTRAL", "S-NEUTRAL",
    "B-CONFLICT", "I-CONFLICT", "E-CONFLICT", "S-CONFLICT"
]

label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# =========================================================
# 2. TOKENIZER
# =========================================================
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# =========================================================
# 3. CONVERT YOUR BIOES DATA -> HF DATASET
# =========================================================
def prepare_dataset(bioes_data):
    tokens_list = []
    labels_list = []

    for sent in bioes_data:
        tokens_list.append([t for t, l in sent])
        labels_list.append([l for t, l in sent])

    return Dataset.from_dict({
        "tokens": tokens_list,
        "labels": labels_list
    })

# =========================================================
# 4. ALIGN LABELS TO BERT TOKENS
# =========================================================
def tokenize_and_align(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    labels_batch = []

    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label2id[word_labels[word_idx]])

            else:
                # subword handling
                label = word_labels[word_idx]
                if label.startswith("I"):
                    label_ids.append(label2id[label])
                else:
                    label_ids.append(-100)

            previous_word_idx = word_idx

        labels_batch.append(label_ids)

    tokenized_inputs["labels"] = labels_batch
    return tokenized_inputs

# =========================================================
# 5. LOAD DATA
# =========================================================
bioes_data = convert(final_df)   
dataset = prepare_dataset(bioes_data)

dataset = dataset.train_test_split(test_size=0.2)

tokenized_dataset = dataset.map(tokenize_and_align, batched=True)

# =========================================================
# 6. MODEL
# =========================================================
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# =========================================================
# 7. TRAINING SETUP
# =========================================================
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./new_absa_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)

# =========================================================
# 8. METRICS
# =========================================================
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_preds = []

    for pred, lab in zip(predictions, labels):
        temp_labels = []
        temp_preds = []

        for p_i, l_i in zip(pred, lab):
            if l_i != -100:
                temp_labels.append(id2label[int(l_i)])
                temp_preds.append(id2label[int(p_i)])

        true_labels.append(temp_labels)
        true_preds.append(temp_preds)

    return {
        "report": classification_report(true_labels, true_preds, digits=4)
    }

# =========================================================
# 9. TRAINER
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================================================
# 10. TRAIN
# =========================================================
trainer.train()


# In[43]:


tokenizer.save_pretrained("./new_absa_model")
model.save_pretrained("./new_absa_model")


# # Inference

# In[44]:


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================================================
# 1. LOAD MODEL + TOKENIZER (NOW CORRECT)
# =========================================================
model_path = "./new_absa_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

model.eval()

# =========================================================
# 2. LABEL MAP (must match training)
# =========================================================
labels = [
    "O",
    "B-POSITIVE", "I-POSITIVE", "E-POSITIVE", "S-POSITIVE",
    "B-NEGATIVE", "I-NEGATIVE", "E-NEGATIVE", "S-NEGATIVE",
    "B-NEUTRAL", "I-NEUTRAL", "E-NEUTRAL", "S-NEUTRAL"
]

id2label = {i: l for i, l in enumerate(labels)}

# =========================================================
# 3. PREDICT TOKEN LABELS
# =========================================================
def predict(sentence):
    tokens = sentence.lower().split()

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    word_ids = inputs.word_ids()

    results = []
    prev_word = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx != prev_word:
            token = tokens[word_idx]
            label = id2label[int(preds[i])]
            results.append((token, label))

        prev_word = word_idx

    return results

# =========================================================
# 4. EXTRACT ASPECTS (BIOES decoding)
# =========================================================
def extract_aspects(predictions):
    aspects = []
    current = []

    for token, label in predictions:

        if label.startswith("B-") or label.startswith("S-"):
            if current:
                aspects.append(current)
            current = [(token, label)]

        elif label.startswith("I-") or label.startswith("E-"):
            current.append((token, label))

        else:
            if current:
                aspects.append(current)
                current = []

    if current:
        aspects.append(current)

    return aspects

# =========================================================
# 5. FORMAT FINAL OUTPUT
# =========================================================
def format_output(aspects):
    results = []

    for asp in aspects:
        words = [w for w, l in asp]
        label = asp[0][1]

        if "POSITIVE" in label:
            sentiment = "POSITIVE"
        elif "NEGATIVE" in label:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        results.append({
            "aspect": " ".join(words),
            "sentiment": sentiment
        })

    return results

# =========================================================
# 6. FULL PIPELINE
# =========================================================
def analyze(sentence):
    preds = predict(sentence)
    aspects = extract_aspects(preds)
    return format_output(aspects)

# =========================================================
# 7. TEST
# =========================================================
if __name__ == "__main__":
    sentence = "The phone has a great camera, but the screen brightness is a bit disappointing."

    print(analyze(sentence))


# In[25]:


final_df.head()


# In[ ]:


final


# # Convert file to .py

# In[205]:


from nbconvert import ScriptExporter
import nbformat

with open("absa-e2e.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

exporter = ScriptExporter()
script, _ = exporter.from_notebook_node(nb)

with open("mypythonfile.py", "w") as f:
    f.write(script)


# In[ ]:





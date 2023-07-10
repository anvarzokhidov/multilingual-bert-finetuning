import transformers
import torch
import pandas as pd
import numpy as np

# Loading dataframes
train = pd.read_csv('my_train.csv', sep=';')  # ; is the delimiter used to separate columns (by default commas are used as delimiters)
test = pd.read_csv('my_test.csv', sep=';')  # ; is the delimiter used to separate columns (by default commas are used as delimiters)

df_train = train.copy()  # making a deep copy in a new memory space
df_test = test.copy()  # making a deep copy in a new memory space

# Encoding the intents target labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df_train['intent'].values)
y_labels_train = le.transform(df_train['intent'].values)
y_labels_test = le.transform(df_test['intent'].values)


############################################# Multilingual Bert ########################################################

def multilingual_bert_finetuning():
    from datasets import load_dataset, load_metric
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
    from transformers import AutoModelForSequenceClassification, get_scheduler
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    # Preparing the datasets from our local machine   
    datasets = load_dataset('csv', data_files={'train': 'my_train.csv', 'test': 'my_test.csv'}, delimiter=';')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True)
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(['text', 'intent'])
    tokenized_datasets['train'] = tokenized_datasets['train'].add_column('labels', y_labels_train) 
    tokenized_datasets['test'] = tokenized_datasets['test'].add_column('labels', y_labels_test)
    tokenized_datasets.set_format('torch')

    train_dataloader = DataLoader(dataset=tokenized_datasets['train'], shuffle=True, batch_size=16, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset=tokenized_datasets['test'], shuffle=True, batch_size=16, collate_fn=data_collator)

    device = torch.device('cuda' if torch.cuda.is_available() else'cpu') # Using GPU

    def model_training():
        # Training
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(y_labels_train))
        for param in model.base_model.parameters():
            param.requires_grad = False # Freezing all the deep layers except the head
        
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=0.0001)
        num_epochs=1
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        progress_bar = tqdm(range(num_training_steps)) # keeping track of the training

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        torch.cuda.empty_cache()
        model.save_pretrained('fine_tuned_models/multilingual_bert/')

        return

    model_training() # calling training function

    model = AutoModelForSequenceClassification.from_pretrained('fine_tuned_models/multilingual_bert_propme/')
    model = model.to(device)
    model.eval()

    # Evaluation -------------------------
    # Accuracy
    metric_accuracy = load_metric('accuracy')
    
    for batch1 in test_dataloader:
        batch1 = {k: v.to(device) for k, v in batch1.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch1['input_ids'], token_type_ids=batch1['token_type_ids'], attention_mask=batch1['attention_mask'], labels=batch1['labels'])
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric_accuracy.add_batch(predictions=predictions, references=batch1['labels'])
    
    accuracy = metric_accuracy.compute()
    print(f'Model accuracy: {accuracy}')
    torch.cuda.empty_cache()


multilingual_bert_finetuning()

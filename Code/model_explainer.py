from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import torch
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import copy as cp
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression


def get_model(modelType = 'Camembert_base'):
    
    modelname = 'camembert-base'
    do_lowercase = ("cased" not in modelname)

    if modelType == "Camembert_base":
        # Load pretrained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(modelname)
        tokenizer = AutoTokenizer.from_pretrained(modelname, do_lowercase=do_lowercase)

    elif modelType == "Camembert_trained":
        # Load trained model and tokenizer trained on preprocessed input
        model = AutoModelForSequenceClassification.from_pretrained("../NeededFiles/camembert_model_punct")
        tokenizer = AutoTokenizer.from_pretrained("../NeededFiles/camembert_model_punct", do_lowercase=do_lowercase)
        
    elif modelType == "Camembert_artifacts":
    # Load trained model and tokenizer trained on preprocessed input
        model = AutoModelForSequenceClassification.from_pretrained("../NeededFiles/camembert_model_artifacts")
        tokenizer = AutoTokenizer.from_pretrained("../NeededFiles/camembert_model_artifacts", do_lowercase=do_lowercase)
    
    elif modelType == "Camembert_allocine":
        # Load pretrained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained('../NeededFiles/camembert_model_allocine')
        tokenizer = AutoTokenizer.from_pretrained(modelname, do_lowercase=do_lowercase)
    
    elif modelType == "Camembert_allocine_full":
        model = AutoModelForSequenceClassification.from_pretrained('../NeededFiles/camembert_model_allocine_full')
        tokenizer = AutoTokenizer.from_pretrained(modelname, do_lowercase=do_lowercase)
    else:
        print("Unknown modelType, choose between Camembert_base, Camembert_clean and Camembert_artifacts")
        print("Exiting")
        sys.exit(0)
    
    return model, tokenizer

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_model(model, train_set, val, device, epochs=2, batch_size=4, seed=42, inplace=False):
    
    if not inplace:
        model = cp.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    batches_per_epoch = len(train_set) // batch_size

    total_steps = int(batches_per_epoch * epochs)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(tqdm(train_set, total=len(train_set))):

            cp_batch = cp.deepcopy(batch)
            b_input_ids = cp_batch['input_ids'].to(device)
            b_input_mask = cp_batch['attention_mask'].to(device)
            b_labels = cp_batch['labels'].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)[:2]

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_set)            

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for step, batch in enumerate(tqdm(val, total=len(val))):

            cp_batch = cp.deepcopy(batch)
            b_input_ids = cp_batch['input_ids'].to(device)
            b_input_mask = cp_batch['attention_mask'].to(device)
            b_labels = cp_batch['labels'].to(device)

            with torch.no_grad():        

                loss, logits = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)[:2]

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)


        avg_val_accuracy = total_eval_accuracy / len(val)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(val)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    return model

def test_model(model, test, device):
    print("")
    print("Running Test...")
    import time 
    import copy as cp
    t0 = time.time()
    total_eval_accuracy = 0

    model.eval()

    predictions , true_labels = [], []

    for step, batch in enumerate(tqdm(test, total = len(test))):

        cp_batch = cp.deepcopy(batch)
        b_input_ids = cp_batch['input_ids'].to(device)
        b_input_mask = cp_batch['attention_mask'].to(device)
        b_labels = cp_batch['labels'].to(device)

        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

        predictions.append(logits)
        true_labels.append(label_ids)


    avg_test_accuracy = total_eval_accuracy / len(test)
    print("  Accuracy: {0:.4f}".format(avg_test_accuracy))

    
def save_model(model, tokenizer, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    
def extract_embeddings(dataset, model, layers = None):
    
    model.eval()
    
    device = next(model.parameters()).device 
    if layers == None:
        layers = [i for i in range(13)]
        
    text_embeddings = {i: [] for i in layers}
    labels = []
    preds = []

    for step, batch in enumerate(tqdm(dataset, total = len(dataset))):

            cp_batch = cp.deepcopy(batch)
            b_input_ids = cp_batch['input_ids'].to(device)
            b_input_mask = cp_batch['attention_mask'].to(device)
            b_labels = cp_batch['labels'].to(device)

            with torch.no_grad():        
                
                outputs = model(b_input_ids, 
                                attention_mask=b_input_mask,
                                token_type_ids = None, 
                                output_hidden_states=True)
                for layer in layers:
                    cls_attn = outputs.hidden_states[layer][:, 0, :].detach().cpu().numpy() 
                    text_embeddings[layer] += [i for i in cls_attn]
                labels += [int(i) for i in b_labels.detach().cpu()]
                preds += [np.argmax(i) for i in outputs[0].detach().cpu().numpy()]
                
    return text_embeddings, labels, preds

def predict_with_embeddings(text_embeddings_train, labels_train, text_embeddings, labels, reduce_dim = False, with_learning_curve = False):

        
    def fit_predict(e_train, lab_train, e):
        
        reg = LinearRegression().fit(e_train, lab_train)
        pre_train = reg.predict(e_train)
        pre = reg.predict(e)
        
        return pre_train, pre, reg
        
    def reduce(e_train, lab_train, e):
        if i*10 <= 2000: #Alternate between 0 (no pca in the beggining) and 2000
            pca = PCA(i)
            pca.fit(e_train)
            embeddings_train_reduce = pca.transform(e_train)
            embeddings_reduce = pca.transform(e)
            
        return embeddings_train_reduce, embeddings_reduce
            
    emb_len = len(list(text_embeddings_train.values())[0])
    nb_step = emb_len//10
    acc = {i: [0 for _ in range(emb_len)] for i in text_embeddings_train}
    all_idx = [i for i in range(emb_len)]
    
    if with_learning_curve:
        for i in tqdm(range(nb_step), total=nb_step):
            idx = np.random.choice(all_idx, (i+1)*10, replace = False)

            for layer in text_embeddings:
                e_train = [text_embeddings_train[layer][j] for j in idx]
                lab_train = [labels_train[j] for j in idx]
                e = text_embeddings[layer]
                
                if reduce_dim:
                    e_train, e = reduce(e_train, lab_train, e)
                
                pre_train, pre, reg = fit_predict(e_train, lab_train, e)
                
                acc[layer][i] = accuracy_score([int(i > 0.5) for i in pre], labels)
                
                
    else:
        for layer in text_embeddings:
            e_train = text_embeddings_train[layer]
            lab_train = labels_train
            e = text_embeddings[layer]
            
            if reduce_dim:
                e_train, e = reduce(e_train, lab_train, e)

            pre_train, pre, reg = fit_predict(e_train, lab_train, e)
            acc[layer] = accuracy_score([int(i > 0.5) for i in pre], labels)
                    

    return pre_train, pre, acc, reg

def normalize(vector, fullExpl=True, sample=False, topP=False, filter=False):
    neg = sum(vector) < 0
    if neg:
        vector = [-i for i in vector]
    
    min_val = min(vector)
    max_val = max(vector)
    
    vector = [(i-min_val)/(max_val-min_val) for i in vector]
    assert max(vector) == 1
    assert min(vector) == 0
    vect_sum = sum(vector)
    vect_len = len(vector)
    
    normalized = [i for i in vector]
    if fullExpl:
        normalized = [i/vect_sum for i in normalized]
    if sample:
        s_norm = sorted([(pos, i) for pos, i in enumerate(normalized)], reverse = True, key = lambda a: a[1])
        sampled = sorted([(initPos, i) if pos < 20 else (initPos, 0) for pos, (initPos, i) in enumerate(s_norm)])
        normalized = [i for pos, i in sampled]
    if topP:
        s_norm = sorted([(pos, i) for pos, i in enumerate(normalized)], reverse = True, key = lambda a: a[1])
        pTot = 0
        vect_sum = sum(normalized)
        sampled = []
        for pos, (initPos, i) in enumerate(s_norm):
            if pTot <= 0.5*vect_sum:
                pTot += i
                sampled.append((initPos, i))
            else:
                sampled.append((initPos, 0))
            
        sampled = sorted(sampled)
        normalized = [i for pos, i in sampled]
    if filter:
        normalized = [i if i>1/len(normalized) else 0 for i in normalized]
    if neg:
        normalized = [-i for i in normalized]

    return normalized

def lime_predictor(texts):
    text_df = pd.DataFrame(texts, columns=['text'])
    textDataset = datasets.Dataset.from_pandas(text_df)
    tokenized_text = textDataset.map(preprocess_function, batched=True)
    
    tokenized_text = tokenized_text.remove_columns(['text', 'special_tokens'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length = max_length, padding = 'max_length')
    text_dl = DataLoader(tokenized_text, 
                   shuffle=False,
                   batch_size=16,
                   collate_fn=data_collator)
    predictions = []
    for step, batch in enumerate(text_dl):
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)

        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        
        predictions += [logit for logit in logits]
        
        b_input_ids.detach().cpu()
        b_input_mask.detach().cpu()
    
    lastLayer = torch.nn.Softmax(dim=1)
    lastLayer.to(device)
    output = lastLayer(torch.tensor(np.array(predictions)))
    lastLayer.cpu()
    return output.cpu().numpy()
import re
import unicodedata
import pandas as pd

def preprocess(text, full = False):
    
    if full:
        for word in text.split(' '):
            if 'http' in word or 'www.' in word or ".be"  in word or ".fr" in word or ".com" in word:
                text = text.replace(word, '<URL>')
    
    
    new_text = unicodedata.normalize('NFKC', text)
    new_text = new_text.replace(u"\u2044", "/")#Fraction bar
    new_text = new_text.replace('\u200b', ' ') #0 width character
    new_text = new_text.replace('\u0009', ' ')#Tabulate
    new_text = new_text.replace('\u200e', '')# Concat letters (ex: oeil)
    new_text = new_text.replace('\u0020', ' ')# No break line
    new_text = new_text.replace('\u202c', ' ')# Pop directional
    new_text = new_text.replace('\u202a', ' ')# Left to right embedding
    new_text = new_text.replace(u'\u25ba', ' ')# Pointeur noir vers la droite
    new_text = new_text.replace('&amp', '&') # Esperluette
    
    #1 Replace the apostrophes
    
    if full:
        #4 Replace the numbers
        nbr_regex = "(?<=[^\w])(?<![a-zA-Z]\-)[0-9]+(?=[^\w])|(?<=[hH])[0-9]+|[0-9]+(?=[hH])"
        nbr_regex2 = "(?=^)[0-9]+"
        new_text = re.sub(nbr_regex, "<NBR>", new_text)
        new_text = re.sub(nbr_regex2, "<NBR>", new_text)


        flag = True
        comp_nbr_regex = "<NBR>[., ]?<NBR>"
        while flag:
            flag = False
            old_text = new_text
            new_text = re.sub(comp_nbr_regex, "<NBR>", new_text)
            flag = (old_text != new_text)

        new_text = new_text.replace("<NBR>", " NBR ")
    
    
    new_text = re.sub('(?<=[^\.])[.](?=[^\.]|$)', ' . ', new_text)
    new_text = re.sub('\]', ' ] ', new_text)
    new_text = re.sub('\[', ' [ ', new_text)
    
    new_text = re.sub('\?', ' ? ', new_text)
    new_text = re.sub('\!', ' ! ', new_text)
    new_text = re.sub(',', ' , ', new_text)
    new_text = re.sub('\;', ' ; ', new_text)
    new_text = re.sub('\(', ' ( ', new_text)
    new_text = re.sub('\)', ' ) ', new_text)
    new_text = re.sub('…|\.\.\.', ' ... ', new_text)
    new_text = re.sub(':', ' : ', new_text)
    new_text = re.sub('#', ' # ', new_text)
    
    #apo_regex = "(\W[cçdjlmnst]|\Wjusqu|\Wqu|[CÇDJLMNST]|Qu|Jusqu)[’'ʼ’‘’ʼ']"
    apo_regex = "(\W[cçdjlmnst]|\Wjusqu|\Wqu|\Wpuisqu|[CÇDJLMNST]|Qu|Jusqu|Puisqu)[’'ʼ’‘’ʼ']"
    characters = re.findall(apo_regex, new_text)
    new_text = re.sub(apo_regex, " <APO0> ", new_text)

    #2 Replace the inword apostrophes (ex: aujourd'hui)
    word_apo_regex = "(?<=[a-z])[’'ʼ’‘’ʼ](?=[a-z])"
    new_text = re.sub(word_apo_regex, " <APO1> ", new_text)
    
    #3 Replace all the other ones by quotes
    quote_regex = "['\"’“”«»‘’ʼ″]"
    new_text = re.sub(quote_regex, " <QUOTE> ", new_text)
    
    
    character_pos = 0
    clean_text = ""
    space_next=True
    for elem in new_text.split():
        elem = elem.strip()
        if elem == "<APO0>":
            clean_text += " " + characters[character_pos].strip() + "'"
            character_pos += 1
        elif elem == "<APO1>":
            clean_text += "'"
            space_next = False
        elif elem == "<QUOTE>":
            clean_text += ' "'
        else:
            if space_next:
                clean_text += " "
            space_next = True
            clean_text += elem

    assert character_pos == len(characters)
    return clean_text.strip()


def create_dataset(dataDfInit, seed = None):
    
    if seed is not None:
        dataDfInit = dataDfInit.sample(frac = 1, random_state = seed)
    
    df_actu = dataDfInit[dataDfInit.label == 0]
    df_chron = dataDfInit[dataDfInit.label == 1]

    for row in df_actu.iterrows():
        idx = row[1][0]
        text = row[1][3]
        i="Liste des films en compétition lors du 65e Festival de Cannes"
        if i in text: 
            df_actu = df_actu.drop(idx)

    df_chron = df_chron.reset_index()
    df_chron = df_chron.loc[df_chron.index < 5000]
    df_chron = df_chron.sample(frac=1, random_state=42).reset_index()

    df_actu = df_actu.reset_index()
    df_actu = df_actu.loc[df_actu.index < 5000]
    df_actu = df_actu.sample(frac=1, random_state=42).reset_index()

    train_chron = df_chron[df_chron.index < 4000]
    val_chron = df_chron[(df_chron.index >= 4000) & (df_chron.index < 4500)]
    test_chron = df_chron[(df_chron.index >= 4500)]

    train_actu = df_actu[df_actu.index < 4000]
    val_actu = df_actu[(df_actu.index >= 4000) & (df_actu.index < 4500)]
    test_actu = df_actu[(df_actu.index >= 4500)]

    train_df = pd.concat([train_chron, train_actu], axis=0).reset_index(drop=True)
    val_df = pd.concat([val_chron, val_actu], axis=0).reset_index(drop=True)
    test_df = pd.concat([test_chron, test_actu], axis=0).reset_index(drop=True)

    
    
    return train_df, val_df, test_df

def convert_to_tok(ids, tokenizer, special_tokens=None):
    retokenized_text = ' '.join(tokenizer.convert_ids_to_tokens(ids))
    
    flag = True
    while flag:
        old_cut = retokenized_text
        retokenized_text = retokenized_text.replace('<unk> ▁ <unk>', '<unk>') 
        flag = (old_cut!=retokenized_text)
        
    retokenized_text = retokenized_text.split()
    count = 0
    for pos, i in enumerate(retokenized_text):
        if i == '<unk>':
            try:
                retokenized_text[pos] = special_tokens[count]
                count += 1
            except:
                print(retokenized_text, special_tokens)
                sys.exit(0)
    
    return retokenized_text
    
def tokenize(textList, tokenizer, shorten=False):
    tokenized = tokenizer(textList, max_length = 512, padding = 'max_length', truncation=True)
    tokenized['special_tokens'] = []
    
    for textNbr, text in enumerate(textList): 
        ids = tokenized['input_ids'][textNbr]
        if shorten:
            count = 1
            while ids[-count] not in [9, 83, 106, 186] and not (ids[-count] == 87 and ids[-count-1] == 9):
                ids[-count] = 1
                tokenized['attention_mask'][textNbr][-count] = 0
                count += 1

            ids[-count+1] = 6
            tokenized['attention_mask'][textNbr][-count+1] = 1
            tokenized['input_ids'][textNbr] = ids
        
        cut_text = ''.join(tokenizer.convert_ids_to_tokens(ids)).replace('<s>▁', '').replace('</s>', '▁').strip().split('▁')
        cut_text = ' '.join(cut_text)
        #join the multiples consecutives unk tokens to avoid multi byte special tokens problems
        flag = True
        while flag:
            old_cut = cut_text
            cut_text = cut_text.replace('<unk> <unk>', '<unk>') 
            flag = (old_cut!=cut_text)
        known_parts = cut_text.split('<unk>')
        if len(known_parts) == 1:
            tokenized['special_tokens'].append([])
            continue
            
        for part in known_parts:
            part = part.replace('<pad>', '').strip()
            if part != '':
                text = text.replace(part, '<SEP>', 1)
        
        special_tokens = text.split('<SEP>')
        special_tokens = [i for i in special_tokens if i != '']
        tokenized['special_tokens'] += [special_tokens]
    return tokenized


def get_text_tokens(text, tokenizer, preprocess_text=False, shorten=True):
    if preprocess_text:
        text = preprocess(text)
    tokenized = tokenize([text], tokenizer, shorten=shorten)
    ids, sp_tok = tokenized['input_ids'][0], tokenized['special_tokens'][0]
    tokens = convert_to_tok(ids, tokenizer, sp_tok)
    correct_tokens = [i.strip() for i in retokenize(tokens, [0 for i in tokens])[0]]
    return ' '.join(correct_tokens).replace('</s>', '')


def retokenize(tokens, expl):

    last = tokens[0]
    sumExpl = expl[0]
    nbrPart = 1
    flagNext = False
    tokenExplList = []

    for token, expli in [(tokens[i], expl[i]) for i in range(len(tokens))]:
        if token[0] != "<" and token[0] != "▁":
            last += token
            sumExpl += expli
            nbrPart += 1
        else:
            tokenExplList += [(last, sumExpl/nbrPart)]
            last = token
            sumExpl = expli
            nbrPart = 1
            if "</s>" in token:
                break

    tokenExplList += [(token, expli)]
    tokenExplList = tokenExplList[2:]
    
    
    correctTokList = "".join([token for token, _ in tokenExplList]).split("▁")
    correctTokList = [token for token in correctTokList if token != '']
    expli = [ex for _, ex in tokenExplList]
    tokensFinal = correctTokList
    return tokensFinal, expli
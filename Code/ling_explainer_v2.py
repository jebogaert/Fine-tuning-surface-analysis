import xlrd
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger
import pandas as pd 
import json
import math
import spacy
import re
import fr_core_news_sm
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

feature_cols_orig = [#"text_length",
                "length_words",
                "cttr",
                "pron_1",
                #"pron_2",
                "nb_on",
                "pron_rel",
                "nb_neg",
                "nb_digits",
                "nb_vb",
                "nb_adj",
                "nb_interrog",
                "nb_exclam",
                "nb_pointvirg",
                "nb_deuxpoints",
                "nb_susp",
                "lexique3_sentiment",
                "nrc_sentiment" ,
                "nb_citations",
                "nb_long_words"]
                #"subjective_frequency",
                #"hapaxes"]

additional_features = ["week_day","discourse_markers","citing_verbs", "concreteness", "imageability", "passives", "thinking_verbs", "subjective_frequency", 'rel_time_markers']
feature_cols_enriched = feature_cols_orig + additional_features

feature_list = feature_cols_enriched


def word_list(vocab):
    dict_word_list = {}
    
    dict_word_list['prons_1'] = ["moi", "je", "me", "nous", "mien", "miens", "nos", "notre", "nôtres", "mon", "ma", "mes", "moi", "nous", "j'", "m'"]
    
    dict_word_list['prons_2'] = ["toi", "tu", "te", "vous", "tien", "tiens", "vos", "votre", "votres", "ton", "ta", "tes", "toi", "vous"]
    
    dict_word_list['pr_rel'] = ["qui", "que", "dont", "où", "lequel", "quoi", "qu'"]
    
    dict_word_list['conj_comp'] = ["or", "donc", "car", "pourtant", "cependant", "néanmoins", "toutefois", "malgré", "outre", "quoique", "tandis"]
    
    dict_word_list['adv_mod'] = ["vraiment", "réellement", "sûrement", "certainement", "assurément", "incontestablement", "sans doute", "sans aucun doute", "heureusement", "malheureusement", "peut-être"]
    
    dict_word_list['negs'] = ["ne", "ni", "n", "non", "aucun", "sans", "nul", "nulle", "n'"]
    
    with open("NeededFiles/lexicons/Time_markers.txt") as f:
        dict_word_list['week_day'] = [line.strip("\n") for line in f] 
    
    with open("NeededFiles/lexicons/Relative_time_markers.txt") as f:
        dict_word_list['rel_time_markers'] = [line.strip("\n") for line in f] 
    
    with open("NeededFiles/lexicons/Citing_verbs.txt") as f:
        dict_word_list['citing_verbs'] = [line.strip("\n") for line in f] 
        
    with open("NeededFiles/lexicons/Thinking_verbs.txt") as f:
        dict_word_list['thinking_verbs'] = [line.strip("\n") for line in f]

    with open("NeededFiles/lexicons/Discourse_markers.txt") as f:
        dict_word_list['discourse_markers'] = [line.strip("\n") for line in f]
        
    dict_word_list["hapaxes"] = [i for i,j in vocab.items() if j==1]

    return dict_word_list


def build_lexicons():
    lexique3_path = f"NeededFiles/lexicons/open_lexicon.csv"
    nrc_path = f"NeededFiles/lexicons/nrc_lexicon.json"

    # create spacy lemmatizer
    @Language.factory('french_lemmatizer')
    def create_french_lemmatizer(nlp, name):
        return LefffLemmatizer(after_melt=True, default=True)

    # create spacy POS-tagger
    @Language.factory('melt_tagger')
    def create_melt_tagger(nlp, name):
        return POSTagger()


    # load Lexique3 lexicon
    columns = ["Word", "Pol.Val.", "Catégories"]
    lexique3_df = pd.read_csv(lexique3_path, sep=';', usecols = columns)
    lexique3_df.set_index("Word", inplace=True)
    lexique3_dict = lexique3_df.to_dict()
    lexique3_dict = lexique3_dict["Pol.Val."]

    # load NRC lexicon
    with open(nrc_path,'r') as f:
        data = json.loads(f.read())

    nrc_df = pd.json_normalize(data, record_path =['LexicalEntry'])
    nrc_df = nrc_df[['Lemma.-writtenForm', 'Sense.Sentiment.-polarity']].copy()
    nrc_df.set_index("Lemma.-writtenForm", inplace=True)
    nrc_df_pos = nrc_df[nrc_df['Sense.Sentiment.-polarity']=="positive"]
    nrc_df_neg = nrc_df[nrc_df['Sense.Sentiment.-polarity']=="negative"]
    nrc_lex_df = pd.concat([nrc_df_pos, nrc_df_neg])
    nrc_dict = nrc_lex_df.to_dict()
    nrc_dict = nrc_dict["Sense.Sentiment.-polarity"]
    
    
    # Bonin, P., Méot, A., & Bugaiska, A. (2018). Concreteness norms for 1,659 French words: Relationships with other psycholinguistic variables and word recognition times. Behavior research methods, 50(6), 2366-2387.
    concreteness_path = "NeededFiles/lexicons/Concr_ContextAv_ValEmo_Arous_1659.xlsx"
    df_concreteness = pd.read_excel(concreteness_path)
    df_concreteness = df_concreteness.drop(index=0)
    concret_dict = dict(zip(df_concreteness['Unnamed: 0'], df_concreteness['Concreteness']))

    # Desrochers, A., & Thompson, G. L. (2009). Subjective frequency and imageability ratings for 3,600 French nouns. Behavior Research Methods, 41(2), 546-557.
    imag_subjfreq_path = "NeededFiles/lexicons/FreqSub_Imag_3600.xls"
    df_imag_subjfreq = pd.read_excel(imag_subjfreq_path)
    imageab_dict = dict(zip(df_imag_subjfreq['Mot'], df_imag_subjfreq['IMAGE_Mean']))
    subjfreq_dict = dict(zip(df_imag_subjfreq['Mot'], df_imag_subjfreq['FREQ_Mean']))
    
    return lexique3_dict, nrc_dict, concret_dict, imageab_dict, subjfreq_dict

def annotate_data(texts):
    spacy.prefer_gpu()
    prefix_re = re.compile(r' ')

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)

    nlp = fr_core_news_sm.load()
    nlp.tokenizer = custom_tokenizer(nlp)
    nlp.add_pipe('melt_tagger', after='parser')
    nlp.add_pipe('french_lemmatizer', after='melt_tagger')

    spacyd_arts_ls = []
    for text in tqdm(texts):
        spacyd_art = nlp(text)
        spacyd_arts_ls.append(spacyd_art)
    
    return spacyd_arts_ls

def get_feature_dict(dict_word_list):
    v_list = ["V", "VINF", "VIMP", "VS", "VERB"]
    feature_dict = {'pron_1': set(dict_word_list['prons_1']),
                'pron_2': set(dict_word_list['prons_2']),
                'nb_on': ['on'],
                'pron_rel': set(dict_word_list['pr_rel']),
                'nb_neg': set(dict_word_list['negs']),
                'nb_digits': ['nbr'],
                'nb_vb': v_list,
                'nb_adj': ['ADJ'],
                'nb_interrog' : ['?'],
                'nb_exclam' : ['!'],
                'nb_pointvirg' : [';'],
                'nb_susp' : ['...'],
                'nb_deuxpoints' : [':'],
                'nb_citations': ['"'],
                'week_day' : set(dict_word_list['week_day']),
                'discourse_markers': set(dict_word_list['discourse_markers']),
                'citing_verbs': set(dict_word_list['citing_verbs']),
                'hapaxes': set(dict_word_list['hapaxes']),
                'passives': ['pass'],
                'rel_time_markers': set(dict_word_list['rel_time_markers'])
                }
    return feature_dict

# Mean concreteness
def concreteness(lemmas, concret_dict):
    
    cnt_found_words = 0
    total_score = 0
    
    for lemma in lemmas:
        if lemma in concret_dict:
            cnt_found_words += 1
            total_score += concret_dict[lemma]
    if cnt_found_words == 0:
        return 0
    return total_score / cnt_found_words

# Mean imageability
def imageability(lemmas, imageab_dict):
    
    cnt_found_words = 0
    total_score = 0
    
    for lemma in lemmas:
        if lemma in imageab_dict:
            cnt_found_words += 1
            total_score += imageab_dict[lemma]
    if cnt_found_words == 0:
        return 0
    return total_score / cnt_found_words

# Mean subjective frequency
def subj_freq(lemmas, subjfreq_dict):
    
    cnt_found_words = 0
    total_score = 0
    
    for lemma in lemmas:
        if lemma in subjfreq_dict:
            cnt_found_words += 1
            total_score += subjfreq_dict[lemma]
    if cnt_found_words == 0:
        return 0
    
    return total_score / cnt_found_words

# Passives
def passives(deps):
    
    cnt_passives = 0
    for dep in deps:
        if dep[-4:] == "pass":
            cnt_passives += 1
    
    return cnt_passives / len(deps)

def hapaxes(lemmas, vocab):
    
    cnt_hapaxes = 0
    for lemma in lemmas:
        if not re.search(r'\d|[A-Z]', lemma):
            if lemma in vocab:
                if vocab[lemma] == 1:
                    cnt_hapaxes += 1
            else:
                cnt_hapaxes += 1
                
    return cnt_hapaxes

def cttr(tokens):

    if len(tokens) == 0:
        return 0
    
    types = {}
    for token in tokens:
        if token not in types:
            types[token] = True
    
    cttr = len(types) / math.sqrt(2 * len(tokens))
    return cttr


def ling_measures(tokens, lemmas, deps, rule_set_list, vocab, dicts):
    measures_dict = {}
    
    counting_dict = {i: 0 for i in feature_list}
    for feature_set in rule_set_list:
        for elem in feature_set:
            if "=" not in elem:
                counting_dict[elem] = counting_dict.get(elem, 0) + 1
    
    for feature, count in counting_dict.items():
        measures_dict[feature] = 100*count/len(tokens) if len(tokens) != 0 else 0
    
    measures_dict["text_length"] =  len(tokens)/512
    measures_dict["length_words"] = sum([len(tok) for tok in tokens])/len(tokens) if len(tokens) != 0 else 0
    measures_dict["cttr"] = cttr(tokens)
    measures_dict["concreteness"] = concreteness(lemmas, dicts["concret_dict"])
    measures_dict["imageability"] = imageability(lemmas, dicts["imageab_dict"])
    measures_dict["subjective_frequency"] = subj_freq(lemmas, dicts["subjfreq_dict"])
    measures_dict["hapaxes"] = hapaxes(lemmas, vocab)
    measures_dict["passives"] = passives(deps)
    
    return measures_dict


def associate_measures(feature_dict, tokens_lc, tags, lemmas, deps, dicts, dict_word_list):
    rule_set_list = [set() for _ in tokens_lc]
    lexique3_dict = dicts['lexique3_dict']
    nrc_dict = dicts['nrc_dict']
    concret_dict = dicts['concret_dict']
    imageab_dict = dicts['imageab_dict']
    subjfreq_dict = dicts['subjfreq_dict']
    
    
    for pos, (tkn, tag, lemma, dep) in enumerate(zip(tokens_lc, tags, lemmas, deps)):
        tkn = tkn.lower()
        for feature, tkn_list in feature_dict.items():
            if feature != 'hapaxes':
                if (tkn in tkn_list) or (tag in tkn_list):
                    rule_set_list[pos].add(feature)
            else:
                if lemma in tkn_list:
                    rule_set_list[pos].add(feature)
        
        if len(tkn) > 8:
            rule_set_list[pos].add("nb_long_words")

        if lemma in lexique3_dict and lexique3_dict[lemma] in ['neg', 'positif']:
            rule_set_list[pos].add("lexique3_sentiment")

        if lemma in nrc_dict and nrc_dict[lemma] in ['negative', 'positive']:
            rule_set_list[pos].add("nrc_sentiment")
        
        if lemma in concret_dict:
            rule_set_list[pos].add(f"concreteness={concret_dict[lemma]}")
            
        if lemma in imageab_dict:
            rule_set_list[pos].add(f"imageability={imageab_dict[lemma]}")
            
        if lemma in subjfreq_dict:
            rule_set_list[pos].add(f"subjective_frequency={subjfreq_dict[lemma]}")

        if lemma in dict_word_list['citing_verbs']:
            rule_set_list[pos].add("citing_verbs")
        
        if lemma in dict_word_list['thinking_verbs']:
            rule_set_list[pos].add("thinking_verbs")
        
        if dep == 'pass':
            rule_set_list[pos].add('passives')
            
    return rule_set_list 


def compute_ling_dataset(spacy_data, feature_dict, dict_word_list, dicts, vocab):
    dict_list = []
    rule_set_lists = []
    
    for spacy_text in tqdm(spacy_data, total=len(spacy_data)):
        text=spacy_text.text
        tokens = text.split(' ')
        tokens_lc = [t.lower() for t in tokens]

        tags = [a.tag_ for a in spacy_text]
        deps = [a.dep_ for a in spacy_text]
        lemmas = [a.lemma_ for a in spacy_text]

        rule_set_list = associate_measures(feature_dict, tokens_lc, tags, lemmas, deps, dicts, dict_word_list)
        rule_set_lists.append(rule_set_list)
        ling_measures_dict = ling_measures(tokens_lc, lemmas, deps, rule_set_list, vocab, dicts) 
        dict_list.append(ling_measures_dict)

            
    ling_dataset = pd.DataFrame.from_dict(dict_list)
    return ling_dataset, rule_set_lists
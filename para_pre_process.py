import os
import nltk
import argparse
import numpy as np
import pandas as pd
import math
from scipy import spatial
from nltk.corpus import brown
from collections import Counter
from punctuator import Punctuator
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('brown')

lmtzr = WordNetLemmatizer()

def takeInput():
    paraInput = ""
    paraInput = input("Enter STT paragraph:")
    return paraInput

def punctPar(paraInput:str):
    for i in range(len(paraInput)):
        if paraInput[i].isalpha():
            break
        else:
            paraInput = paraInput[1:]
    p = Punctuator('Demo-Europarl-EN.pcl')
    return p.punctuate(paraInput)


def get_tag_info(data):

    feature_set = dict()
    ttr = {}

    # columns = [ 'Category', 'ttr', 'R', 'num_concepts_mentioned',  'ARI', 'CLI',
    #            'prp_count', 'prp_noun_ratio', 'Gerund_count', 'NP_count', 'VP_count', 'word_sentence_ratio', 'MLU',
    #            'count_pauses', 'count_unintelligible', 'count_trailing', 'count_repetitions', 'SIM_score', 'Bruten']

    noun_list = ['NN', 'NNS', 'NNP', 'NNPS']
    verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']


    #no. of unique concepts described
    cookie_pic_list = ['cookie','jar','stool','steal', 'sink','kitchen', 'window','curtain','fall']
    list1 = ['mother','woman','lady']
    list2 = ['girl','daughter','sister']
    list3 = ['boy','son','child','kid','brother']
    list4 = ['dish','plate','cup']
    list5 = ['overflow','spill','running']
    list6 = ['dry','wash'] 
    list7 = ['faucet'] 
    list8 = ['counter','cabinet'] 
    list9 = ['water']

    grammar = r"""
    DTR: {<DT><DT>}
    NP: {<DT>?<JJ>*<NN.*>} 
    PP: {<IN><NP>} 
    VPG: {<VBG><NP | PP>}
    VP: {<V.*><NP | PP>}     
    CLAUSE: {<NP><VP>} 
    """  

    # ---------- Distribution feature ----------    
    text = brown.words(categories='news')
    tag_info = nltk.pos_tag(text)
    tag_fd = nltk.FreqDist(tag for (word, tag) in tag_info)
    del_key = []
    for key in tag_fd.keys():
        if not key.isalpha():
            del_key.append(key)
    while not (del_key == []):
        tag_fd.pop(del_key.pop(), None)

    POS_tag = ['NN', 'IN', 'DT', 'VBD', 'VBFG', 'VBG', 'PRP', 'JJ', 'NNP', 'RB', 'NNS', 'CC']        
    tot_pos = sum([tag_fd[tag] for tag in POS_tag])  #sum(tag_fd.values())
    
    global_pos_vec = []
    for tag in POS_tag:
        if tag in list(tag_fd.keys()):
            global_pos_vec.append(tag_fd[tag]/tot_pos)
        else:
            global_pos_vec.append(0)

    # ---------------------tagging information -------------------
        content = data
        text = nltk.word_tokenize(content)    
        
        # ========= LEXICOSYNTACTIC FEATURES =========
        
        #  ------- POS tagging ------- 
        tag_info = np.array(nltk.pos_tag(text))
        tag_fd = nltk.FreqDist(tag for i, (word, tag) in enumerate(tag_info))
        freq_tag = tag_fd.most_common()
        
        # ------- Lemmatize each word -------    
        text_root = [lmtzr.lemmatize(j) for indexj, j in enumerate(text)]
        for indexj, j in enumerate(text):
            if tag_info[indexj,1] in noun_list:
                text_root[indexj] = lmtzr.lemmatize(j) 
            elif tag_info[indexj,1] in verb_list:
                text_root[indexj] = lmtzr.lemmatize(j,'v')             
        
        # ------- Phrase type ------- 
        sentence = nltk.pos_tag(text)
        cp = nltk.RegexpParser(grammar)
        phrase_type = cp.parse(sentence)  
        
        # ------- Pronoun frequency -------
        prp_count = sum([pos[1] for pos in freq_tag if pos[0]=='PRP' or pos[0]=='PRP$'])
        
        # ------- Noun frequency -------
        noun_count = sum([pos[1] for pos in freq_tag if pos[0] in noun_list])
        
        # ------- Gerund frequency -------
        vg_count = sum([pos[1] for pos in freq_tag if pos[0]=='VBG'])
        
        # ------- Pronoun-to-Noun ratio -------
        if noun_count != 0:
            prp_noun_ratio = prp_count/noun_count
        else:
            prp_noun_ratio = prp_count
        
        # Noun phrase, Verb phrase, Verb gerund phrase frequency        
        NP_count = 0
        VP_count = 0
        VGP_count = 0
        for index_t, t in enumerate(phrase_type):
            if not isinstance(phrase_type[index_t],tuple):
                if phrase_type[index_t].label() == 'NP':
                    NP_count = NP_count + 1
                elif phrase_type[index_t].label() == 'VP': 
                    VP_count = VP_count + 1
                elif phrase_type[index_t].label() == 'VGP':
                    VGP_count = VGP_count + 1
                            
        # ------- TTR type-to-token ratio ------- 
        numtokens = len(text)
        freq_token_type = Counter(text)  # or len(set(text)) # text_root
        v = len(freq_token_type)
        ttr = float(v)/numtokens       
                   
        # ------- TTR type-to-token ratio lemmatized------- 
        freq_lemmtoken_type = Counter(text_root)  # or len(set(text)) # text_root
        vl = len(freq_lemmtoken_type)
        ttr_lemmatized = float(vl)/numtokens                         
        
        # ------- Honore's statistic ------- 
        freq_token_root = Counter(text_root)
        occur_once = 0
        for j in freq_token_root:
            if freq_token_root[j] == 1:
                occur_once = occur_once + 1
        v1 = occur_once
        R = 100 * math.log(numtokens / (1 - (v1/v)))
                
        # ------- Automated readability index ------- 
        num_char = len([c for c in content if c.isdigit() or c.isalpha()])
        num_words = len([word for word in content.split(' ') if not word=='' and not word=='.'])
        num_sentences = content.count('.') + content.count('?')
        ARI = 4.71*(num_char/num_words) + 0.5*(num_words/num_sentences) - 21.43
        
        # ------- Coleman–Liau index -------
        L = (num_char/num_words)*100
        S = (num_sentences/num_words)*100
        CLI = 0.0588*L - 0.296*S - 15.8                
            
        # ------- Word-to-sentence_ratio -------
        word_sentence_ratio = num_words/num_sentences
        

        # ========= SEMANTIC FEATURES =========
        
        # ------- Mention of key concepts ------- 
        num_concepts_mentioned = len(set(cookie_pic_list) & set(freq_token_root)) \
                                + len(set(list1) & set(freq_token_root)) + len(set(list2) & set(freq_token_root)) \
                                + len(set(list3) & set(freq_token_root)) + len(set(list4) & set(freq_token_root)) \
                                + len(set(list5) & set(freq_token_root)) + len(set(list6) & set(freq_token_root)) \
                                + len(set(list7) & set(freq_token_root)) + len(set(list8) & set(freq_token_root)) \
                                + len(set(list9) & set(freq_token_root))           								
        
        
        # ---------- Distribution feature ----------    

        sim_score = 0.75

        # ---------- Bruten Index ----------    
        bruten = float(vl)**(numtokens**-0.0165)
        

    extracted_features = [
        'ttr', 'R', 'num_concepts_mentioned', 'ARI', 'CLI', 'prp_count', 'prp_noun_ratio', 'vg_count', 'NP_count', 'VP_count', 
                            'word_sentence_ratio','sim_score', 'bruten']

    for f in extracted_features:
        feature_set[f] = eval(f)

    return feature_set


def main(paraInput):
    print('IN PARA_PROCESS')
    # paraInput = takeInput()
    punctPara = punctPar(paraInput)
    feature_set = get_tag_info(punctPara)
    # print(feature_set)
    usedFeatures =  [feature_set['ttr'],feature_set['R'],feature_set['num_concepts_mentioned'],feature_set['ARI'],feature_set['CLI'],feature_set['prp_count'],feature_set['prp_noun_ratio'],feature_set['NP_count'],feature_set['VP_count'], feature_set['word_sentence_ratio']]
    print(usedFeatures)
    return usedFeatures
    
if __name__ == '__main__':
    feature_set = main(takeInput())

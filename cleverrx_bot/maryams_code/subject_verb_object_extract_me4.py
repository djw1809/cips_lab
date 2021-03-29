#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Issues:
# 1- currently does not capture pobj in active voice example I added milk to flower, flower is not captured ('I', 'added', 'miilk')
# 2- it does not capture the correct sentiment for cases like I saved money by using my cleverrx card

# This version includes the adverb exctraction, append_callexpand function, lexicon map and score tuples
import spacy
import pandas as pd


# use spacy small model
nlp = spacy.load('en_core_web_lg')

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}

# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd", "pobj"}

# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}

# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}
  

# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False



# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v



# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return toks


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited): # sub, tokens, visited
    if item.lower_ == 'that':
        item = _get_that_resolution(tokens)

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    return ' '.join([item.text for item in tokens])

# finds the adverbs that are attached to a verb 
def find_adverbs_of_v(verb):
    print("find_adverbs_of_v:", verb)
    expanded_v = [verb]
    is_expanded = False
    if hasattr(verb, 'lefts'):
        lefts = list(verb.lefts)
        for tok in lefts[-3:]:
            if tok.dep_ in ["advcl", "advmod"]:
                expanded_v = [tok] + expanded_v
                is_expanded = True
                
    if hasattr(verb, 'rights'):
        rights = list(verb.rights)
        for tok in rights[:3]:
            if tok.dep_ in ["advcl", "advmod"]:
                expanded_v = expanded_v + [tok]
                is_expanded = True

    return is_expanded, expanded_v

# reads the lexicon xlsx file and creates a word sentiment map
def create_word_sentiment_map():
    
    xls = "/home/maryam/sentiment_analysis/final_lexicon.xlsx"
    df = pd.read_excel(xls, index=False)
    word_score_lexicon_dict = {}
    for i in range(len(df)):
        word_score_lexicon_dict[df.iloc[i,0]] = df.iloc[i,1]
    return word_score_lexicon_dict

# Given a sub, verb, obj it calculates a score of each phrase
# The score of words are added up together
# In case of verb negation or object negation, the score will multiplied by -1
def calculate_sentiment_score (sub, verb, obj):
    #calculates the sentiment score of the (subject, verb, object)
    word_sentiment_map = create_word_sentiment_map()
    sub_score = 0
    for tok in sub:
        if tok.lemma_ in word_sentiment_map.keys():
            sub_score += word_sentiment_map[tok.lemma_] 
    
    verb_score = 0
    for tok in verb:
        if tok.lemma_ in word_sentiment_map.keys():
            verb_score +=  word_sentiment_map[tok.lemma_]
    if verbNegated or objNegated:
        verb_score = -1*verb_score
   
    obj_score = 0
    for tok in obj:
        if tok.lemma_ in word_sentiment_map.keys():
            obj_score += word_sentiment_map[tok.lemma_]
            
#     if verb.lemma_ in word_sentiment_map.keys():
#         verb_score +=  word_sentiment_map[verb.lemma_]
#     if verbNegated or objNegated:
#         verb_score = -1*verb_score
    return (sub_score, verb_score, obj_score)

# calles the expan function and find adverb function for given sub, verb and obj
# appends the list of (sub, obj, verb) tuples and scores
def append_and_callexpand(sub, verb, obj):
    subject_extended = expand(sub, tokens, visited)
    object_extended = expand(obj, tokens, visited)
    is_expanded, expanded_v = find_adverbs_of_v(verb)
    print("append_and_callexpand:", is_expanded, to_str(expanded_v))
    if is_expanded:
        verb_to_append_list = []
        for item in expanded_v:
            if item==verb:
                verb_to_append_list.append("!" + verb.lemma_ if verbNegated or objNegated else verb.lemma_)
    #             .join("!" + verb.lemma_ if verbNegated or objNegated else verb.lemma_)
            else:
                verb_to_append_list.append(item.lemma_)
    #             verb_to_append = ' '.join(item.lemma_)
        verb_to_append = ' '.join(item for item in verb_to_append_list)
    else:
        verb_to_append = "!" + verb.lemma_ if verbNegated or objNegated else verb.lemma_ # in active voices, v.lower_ can be used
    
    svos.append((to_str(subject_extended), verb_to_append, to_str(object_extended)))
    svos_sentiment_score.append(calculate_sentiment_score(subject_extended, expanded_v, object_extended))
    return
            

# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    global verbNegated
    global objNegated
    global svos
    global svos_sentiment_score
    global visited
    svos = []
    svos_sentiment_score = []
    is_pas = _is_passive(tokens)
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    visited = set()  # recursion detection
    for v in verbs:
#         print("we are in the main vfor loop:", v)
        subs, verbNegated = _get_all_subs(v)
#         print(v, "subs, verbNegated" , subs, verbNegated)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
#             print("len(subs) > 0")
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
#             print('isConjVerb, conjV', isConjVerb, conjV)
            if isConjVerb:
#                 print("if isConjVerb:", sub, v, obj)
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        
                        if is_pas:  # reverse object / subject for passive
                            append_and_callexpand(obj, v, sub)
                            append_and_callexpand (obj, v2, sub)
                        else:
                            append_and_callexpand(sub, v, obj)
                            append_and_callexpand(sub, v2, obj)               
            else:
#                 print("else of isConjVerb:", v, subs)
                v, objs = _get_all_objs(v, is_pas)
#                 print("before for inside else",subs, v, objs)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)

                        if is_pas:  # reverse object / subject for passive
                            append_and_callexpand(obj, v, sub)

                        else:
                            print("right before append stage:", subs, v, objs)
                            append_and_callexpand(sub, v, obj)

    return svos, svos_sentiment_score


# In[8]:


from spacy import displacy
tokens = nlp("I hate sympathy and flower don't like lovely happiness")
for sent in tokens.sents:
    print(sent.text,"\n")

print(type(' '.join([item.text for item in tokens])))
# print(tokens[1].dep_)
for token in tokens:
    print(token.text, "->", token.tag_ ,"->", token.pos_, "->", token.dep_ , "->", token.orth_ , "->",token.lemma_,"" )

svos, svos_sentiment_score = findSVOs(tokens)
print(svos)
print(svos_sentiment_score)
# displacy.serve(tokens, style='dep')

# In[ ]:




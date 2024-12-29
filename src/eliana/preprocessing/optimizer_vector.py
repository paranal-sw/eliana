import numpy as np
from sklearn.cluster import DBSCAN
import logging
from collections import Counter

from .utils import get_logvocab



def optimized_tkn(df, tknClass, column='logtext', warn=False):
    if isinstance(warn, bool) and warn:
        log = logging.warning
    else:
        log = lambda x: x


    log("### Step 1: Basic tokenization")

    tkn=tknClass()
    df['tkn_simple'] = df[column].apply(tkn.tokenize)

    log( f"There are {len(df)} event logs in the trace" )

    # Randomize to avoid artifacts
    df=df.sample(frac=1).reset_index(drop=True)

    vocab = get_logvocab(df['tkn_simple'])
    color_count = vocab['color_count']
    df_colors = vocab['df_colors']
    
    log(f'There are {len(color_count.keys())} unique colors')

    log("### Step 2: Extract low frequency colors. ")

    THRESHOLD_LOW_F_COLOR = np.median(list(color_count.values())) + 1

    df_low_F = df_colors[df_colors['frequency']<=THRESHOLD_LOW_F_COLOR].sort_values('tokenized event')
    df_low_F['TFV_size'] = df_low_F['tokenized event'].apply(lambda x: len( x.split() ))


    log(f'Low Frequency Colors: {len(df_low_F)}')    


    log("### Step 3: Token Frequency Vectors (TFV)")

    # Count colors per each TFV_size
    df_low_size = df_low_F.groupby('TFV_size')[['tokenized event']].count()

    # Count the tokens inside the same TFV_size
    token_count = {}
    for k in df_low_size.index:
        token_count[k] = Counter()
        df_tokens_in_size_k = df_low_F[ df_low_F['TFV_size'] == k ]
        token_count[k].update(' '.join(df_tokens_in_size_k['tokenized event'].values).split( ))

    # Add TFV to each row of low frequency colors
    df_low_F['TFV'] = df_low_F.apply(lambda row: compute_token_vector(row, token_count), axis=1)


    log("### Step 4: Build a Token Frequency Matrix")
    TFM = { size: df_low_F[df_low_F['TFV_size']==size] for size in token_count.keys() }


    log("### Step 5: Clusterize and replace rare events")
    new_templates = []

    for k in df_low_size.index:
        M = TFM[k] 

        # Magic parameters! Needs explanation. 4 is to discard false clusters and have a minimum number of elements.
        X=np.vstack(M['TFV'].to_numpy())
        dbscan = DBSCAN(eps=3, min_samples=4)

        # Fit the model to the data and obtain cluster labels
        labels = dbscan.fit_predict(X)

        M['labels'] = dbscan.labels_

        # Length of every Label
        lenL = { label: len(M[M['labels'] == label]) for label in dbscan.labels_ }

        ### Step 6: Replace rares
        # log(f"### Step 6: Replace rares for TFV size={k}")

        M['TFV2'] = M.apply(lambda row: replace_rare(row, lenL), axis=1)
        M['template'] = M.apply(templatizer, axis=1)
        templates=list(M[M['template']!='']['template'].unique())


        ### Step 6.1: Tokenize common parts
        # log(f"### Step 6.1: Tokenize common parts for TFV size={k}")
        replace={}

        # 'getting header from vlti at § § tcshdr_dcs_§'
        for template in templates:
            replace[template] = {}

            # [5, 6, 7] positions of §
            indexes = [i for i, word in zip( range(len(template)), template.split(' ') ) if word=='§']

            #Parameter: Minimum amount of different paramneters to be considereded for placeholder
            MIN_PARAM_POPULATION = 3

            for i in indexes:
                param = M[M['template']==template]['tokenized event'].apply(lambda x: x.split(' ')[i]).unique()
                # Ommit ['exposre' 'exposure'] and ['start' 'end'], leave ['tcshdr_dcs_102' 'tcshdr_dcs_103' 'tcshdr_dcs_104'...]
                if len(param) >= MIN_PARAM_POPULATION:
                    replace[template][i] = common_head(param) + '§' + common_tail(param)

        M['common_token'] = M.apply(lambda row: common_token(row, replace), axis=1)

        new_templates += list(M[ (M['common_token']!='') & (M['common_token'].str.contains('§')) ]['common_token'].unique())


        # new_templates += list(M[M['template']!='']['template'].unique())

    tkn_vectorized=tknClass()
    tkn_vectorized.templates = new_templates

    return tkn_vectorized



def compute_token_vector(row, token_count):
    """
    Given a string, create a vector with the frequency of every token

    example: "seq ft robj name ngc3603 b"
    vector : [7771, 9306, 153, 8312, 8, 4], "seq ft robj name § §"
    """
    color = row['tokenized event']
    k = row['TFV_size']
    token_vector = color.split(' ')
    freq_vector = [ token_count[k][t] for t in token_vector ]
    return freq_vector    


def replace_rare(row, lenL):
    """
    In each token vector row = Xi = [ xi1, xi2, ..., xij ] where xij is the frequency of token at position j in the event j
    if xij < i = len(L) then replace xij by -1
    """
    vect = row['TFV']
    newvect = []
    L = row['labels']
    for v in vect:
        if v < lenL[L]:
            newvect.append( -1 )
        else:
            newvect.append( v )
    return newvect

def templatizer(row):
    """
    Parameter: max § <= |original|//2
    """
    original = row['tokenized event'].split(' ')
    if row['labels'] == -1:
        template = ''
    else:
        tpl = []
        for i in range(row['TFV_size']):
            if row['TFV2'][i] == -1:
                tpl.append('§')
            else:
                tpl.append(original[i])
        if tpl.count('§') <= len(tpl) // 2:
            template = ' '.join(tpl)
        else:
            template = ''
    return template


def common_head(list_str):
    """
    For ['tcshdr_dcs_102' 'tcshdr_dcs_103' 'tcshdr_dcs_104'...]
    Return 'tcshdr_dcs_§'
    """
    i = 0
    common = ''
    while all([len(x) >= i for x in list_str]) and all([x[i]==list_str[0][i] for x in list_str[1:]]):
        common += list_str[0][i]
        i += 1
    return common

def common_tail(list_str):
    """
    See common_head
    """
    i = 1
    common = ''
    while all([len(x) >= i for x in list_str]) and all([x[-i]==list_str[0][-i] for x in list_str[1:]]):
        common = list_str[0][-i] + common
        i += 1
    return common

def common_token(row, replace):
    """
    Re tokenize using common head and tails
    """
    list_common_token = []
    list_raw = row['tokenized event'].split(' ')
    template = row['template']
    if template == '':
        return ''
    for i, x in zip(range(len(list_raw)), list_raw):
        if i not in replace[template].keys():
            list_common_token.append(x)
        else:
             list_common_token.append(replace[template][i])

    return ' '.join(list_common_token)

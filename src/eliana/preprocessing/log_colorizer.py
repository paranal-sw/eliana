"""
References

1. https://huggingface.co/docs/tokenizers/pipeline

Normalizer : clean up text. 

"""
import re
import pickle
import logging
import copy
import traceback
from types import SimpleNamespace
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from .tokenizers import VltTokenizer, RegExpTokenizer


class LogColorizer():
    """
    Model to learn tokenizers from a dataset of traces.

    Attributes
    ----------
    _special : str
        Placeholder symbol used in templates. Default is "§".
    _wildcard : str
        Placeholder for variable text in templates. Default is "{}".
    _templates : list or None
        Stores the templates created from tokenized traces.
    """
    _special = "§"
    _wildcard = "{}"
    _templates = None

    def __init__(self, tokenizer=None, tokenizer_func = None) -> None:
        """
        Initializes the LogColorizer with a tokenizer or a custom tokenization function.

        Parameters
        ----------
        tokenizer : object, optional
            Tokenizer instance with a `tokenize` method.
        tokenizer_func : callable, optional
            Custom tokenization function.
        """
        if tokenizer:
            # self.regexps = tokenizer.regexps if tokenizer else lambda x: x
            self._tokenizer_func = tokenizer.tokenize
            self.tokenizer_instance = tokenizer
        elif tokenizer_func:
            # self.regexps = tokenizer.regexps if tokenizer else lambda x: x
            self._tokenizer_func = tokenizer.tokenize
            self.tokenizer_instance = None
        else:
            # self.regexps = None
            self._tokenizer_func = lambda x: x
            self.tokenizer_instance = None
        self._post_regexp = RegExpTokenizer()
        self.templates = []

    @property 
    def templates(self):
        """
        List of templates used for matching traces.
        """        
        return self._templates
    
    @templates.setter
    def templates(self, v):
        self._templates = v
        self._templates_pieces = [ v.split() for v in v ]

    @property
    def regexps(self):
        """
        List of regular expressions used by the tokenizer, including post-processing patterns.

        Returns
        -------
        list of str
            Combined list of tokenizer and post-processor regex patterns.
        """        
        x = copy.copy(self.tokenizer_instance.regexps)
        x.append(self._post_regexp.regexps)
        return x

    @property
    def special(self):
        """
        Symbol used as a placeholder for numeric or variable values in templates. Default is '§'.
        """
        return self._special
    
    @property
    def wildcard(self):
        return self._wildcard

    def tokenize(self, word):
        """
        Tokenizes a word or phrase using the provided tokenizer and post-processing steps.

        Parameters
        ----------
        word : str
            Input string to tokenize.

        Returns
        -------
        str
            Tokenized representation of the input.
        """
        txt = self._tokenizer_func(word)
        txt = self._post_regexp.tokenize(f" {txt} ").strip()
        txt = self._templatize(txt)
        return txt

    
    def fit(self, traces, warn=False):
        """
        Learns templates and tokenization rules from a dataset of traces.

        Parameters
        ----------
        traces : pandas.DataFrame
            Dataset containing traces with 'event' and 'trace_id' columns.
        warn : bool, optional
            If True, logs warnings during the process. Default is False.

        Returns
        -------
        pandas.DataFrame
            Processed traces with added tokenized columns and metadata.
        """        
        if isinstance(warn, bool) and warn:
            log = logging.warning
        else:
            log = lambda x: x
        self._post_regexp = RegExpTokenizer()

        log(f"Unique colors with no tokenization = {len(set(traces['event']))}")

        log("Pass 1/4: Raw Tokenizer")

        events = traces.copy()#[['event']].copy()
        events['pass_1'] = events['event'].apply( self._tokenizer_func )
        log(f"Pass 1/4: -- Done! {_evaluate(events['event'], events['pass_1'])}")


        log("Pass 2/4: Parametrize numbers of low freq tokens")

        token_count = Counter(' '.join(events['pass_1']).split(' '))
        df_tokens=pd.DataFrame({ 'token': token_count.keys(), 'frequency': token_count.values() })

        FREQ = max( df_tokens['frequency'].median(), 1+ int(0.01* len(df_tokens)) )

        df_low_freq=df_tokens[ df_tokens['frequency'] <= FREQ ].sort_values(by='token')

        # Replace all numbers by §, and discard no numeric 
        df_no_numbers = df_low_freq['token'].apply(lambda x: re.sub(r'[0-9]+', self.special, x) )
        df_no_numbers = df_no_numbers[ df_no_numbers.str.contains(self.special) ]
        log(f"Pass 2/4: -- # of low freq tokens: {len(df_no_numbers)}")

        pattern_list = [ re.escape(k) for k, v in Counter( df_no_numbers ).items() ]
        for pattern in pattern_list:
            raw_pattern = re.escape(pattern).replace(self.special, "[0-9]+")
            self._post_regexp.add_regexp( f" {raw_pattern} ", r" " + pattern.replace(self.special, "{}") + r" " )

        events['pass_2'] = events['event'].apply( self.tokenize )
        log(f"Pass 2/4: -- Done! {_evaluate(events['event'], events['pass_2'])}")
        

        log("Pass 3/4: Remove numbers in similar alphanum tokens")

        pass_2_tokens = Counter(' '.join(events['pass_2']).split( ))
        df_tokens=pd.DataFrame({ 'alphanum':pass_2_tokens.keys(), 'alphanum_f': pass_2_tokens.values() }).sort_values('alphanum_f')
        df_tokens['alpha'] = df_tokens['alphanum'].apply(lambda x: re.sub(r"[0-9]+", self.special, x))

        df_new_tokens = df_tokens[ df_tokens['alpha'] != df_tokens['alphanum'] ]

        MIN_REPEATING_FREQ = 5

        df_repeated = df_new_tokens.groupby("alpha").count()
        df_repeated = df_repeated[ df_repeated['alphanum'] >= MIN_REPEATING_FREQ ]
        log(f"Pass 3/4: -- Repeated alphanum tokens with length >= {MIN_REPEATING_FREQ}:")
        log(f"Pass 3/4: -- {list(df_repeated.index)}")

        for pattern in df_repeated.index:
            raw_pattern = re.escape(pattern).replace(self.special, "[0-9]+")
            self._post_regexp.add_regexp( f" {raw_pattern} ", r" " + pattern.replace(self.special, self.wildcard) + r" " )
        events['pass_3'] = events['event'].apply( self.tokenize )
        log(f"Pass 3/4: -- Done! {_evaluate(events['event'], events['pass_3'])}")


        log("Pass 4/4: Token Frequency Vector (TFV) optimization")
        df = events['pass_3']

        color_count = Counter(df)
        df_colors=pd.DataFrame({ 'tokenized event':color_count.keys(), 'frequency': color_count.values() }).sort_values('frequency')

        # THRESHOLD_LOW_F_COLOR = max(df_colors['frequency'])/2+1
        THRESHOLD_LOW_F_COLOR = np.median(df_colors['frequency'])+1
        log(f"Pass 4/4: -- THRESHOLD_LOW_F_COLOR = {THRESHOLD_LOW_F_COLOR}")

        df_low_F = df_colors[df_colors['frequency']<=THRESHOLD_LOW_F_COLOR].sort_values('tokenized event')
        df_low_F['TFV_size'] = df_low_F['tokenized event'].apply(lambda x: len( x.split() ))
        log(f"Pass 4/4: -- Number of Low Frequency Colors: {len(df_low_F)}")    
        
        log("Pass 4/4: -- Generate the Token Frequency Vectors (TFV)")

        # Count colors per each TFV_size
        df_low_size = df_low_F.groupby('TFV_size')[['tokenized event']].count()

        token_count = {}
        for k in df_low_size.index:
            token_count[k] = Counter()
            df_tokens_in_size_k = df_low_F[ df_low_F['TFV_size'] == k ]
            token_count[k].update(' '.join(df_tokens_in_size_k['tokenized event'].values).split( ))

        # Add TFV to each row of low frequency colors
        df_low_F['TFV'] = df_low_F.apply(lambda row: compute_token_vector(row, token_count), axis=1)

        # Build a Token Frequency Matrix
        TFM = { size: df_low_F[df_low_F['TFV_size']==size] for size in df_low_F['TFV_size'].unique() }

        log("Pass 4/4: -- Clusterize by TFV_size and replace rare events")
        new_templates = []


        for k, M in TFM.items():

            M['labels'] = cluster_labels(M['TFV'])

            # Length of every Label
            lenL = { label: len(M[M['labels'] == label]) for label in M['labels'] }

            M['TFV2'] = M.apply(lambda row: replace_rare(row, lenL), axis=1)
            M['template'] = M.apply(templatizer, axis=1)
            templates=list(M[M['template']!='']['template'].unique())

            replace={}

            # 'getting header from vlti at § § tcshdr_dcs_§'
            for template in templates:
                replace[template] = {}

                # [5, 6, 7] positions of §
                indexes = [i for i, word in zip( range(len(template)), template.split(' ') ) if word=='§']

                #Parameter: Minimum amount of different paramneters to be considereded for placeholder
                MIN_PARAM_POPULATION = 3

                for i in indexes:
                    param = M[M['template']==template]['tokenized event'].astype(str).apply(lambda x: x.split(' ')[i]).unique()
                    # Ommit ['exposre' 'exposure'] and ['start' 'end'], leave ['tcshdr_dcs_102' 'tcshdr_dcs_103' 'tcshdr_dcs_104'...]
                    if len(param) >= MIN_PARAM_POPULATION:
                        replace[template][i] = common_head(param) + '§' + common_tail(param)

            M['common_token'] = M.apply(lambda row: common_token(row, replace), axis=1)


            new_templates += list(M[ (M['common_token']!='') & (M['common_token'].str.contains('§')) ]['common_token'].unique())

        self.templates = new_templates

        events['pass_4'] = events['event'].apply( self.tokenize )
        stats = _evaluate(events['event'], events['pass_4'])
        log(f"Pass 4/4: Done! { stats }")

        self.colors_ = ['<unk>'] + list(events['pass_4'].unique())
        # self.vocab_dict_= {"<unk>": 0}
        # self.vocab_dict_.update({v: k+1 for k, v in zip( range(len(self.colors_)), self.colors_ ) })
        self.vocab_dict_ = {v: k for k, v in zip( range(len(self.colors_)), self.colors_ ) }

        events['color_id'] = events['pass_4'].apply( lambda x: self.vocab_dict_.get(x, -1) )
        self.traces_ = events.groupby('trace_id')['color_id'].apply(list).to_dict()


        log("------- Results -------")
        log(f"   Templates found: {len(self.templates)}")
        log(f"   Regexps found: {len(self._post_regexp.regexps)}")
        log(f"   Vocabulary Size: {len(self.vocab_dict_)} (unique colors)")
        log(f"   Compression: {stats['compression']}")
        log("Finished!")

        return events


    def _templatize(self, txt):
        """
        Matches a text string against known templates to standardize it.

        Parameters
        ----------
        txt : str
            Text to be matched against templates.

        Returns
        -------
        str
            Standardized text matching a template.
        """        
        txt_pieces = txt.split()
        txt_size = len(txt_pieces)

        CONTINUE_NEXT_TPL = True
        for tpl, tpl_id in zip(self.templates, range(len(self.templates))):
            if CONTINUE_NEXT_TPL:

                #logging.debug(f"Test template '{tpl}'")
                OMMIT_TPL = False

                # Sanity Check: template size must be <= txt
                if not len(tpl) <= len(txt):
                    #logging.debug(f"STOP, not {len(tpl)} <= {len(txt)}")
                    OMMIT_TPL = True

                # Optimization: avoid split tpl every time
                tpl_pieces = self._templates_pieces[tpl_id]
                tpl_size = len(tpl_pieces)

                # Sanity check: chunks must have same size.
                if tpl_size != txt_size:
                    #logging.debug(f"chunks must have same size {tpl_size} != {txt_size}")
                    #logging.debug(tpl_pieces)
                    #logging.debug(txt_pieces)
                    OMMIT_TPL = True

                if not OMMIT_TPL:
                    new_txt = []
                    for tpl_piece, txt_piece in zip(tpl_pieces, txt_pieces):
                        if not OMMIT_TPL:
                            #logging.debug(f"Next chunk: {tpl_piece} / {txt_piece}")
                            if self.special not in tpl_piece:
                                # Sanity check: tpl / txt should be equal piece by piece
                                if tpl_piece != txt_piece:
                                    #logging.debug(f"STOP, No special, but '{tpl_piece}' != '{txt_piece}'")
                                    OMMIT_TPL=True
                                else:
                                    new_txt.append(txt_piece)
                                    #logging.debug(f"Adding equal text = {txt_piece}")

                            # Here is the tricky part...
                            else:
                                #logging.debug("Templatizing chunk")
                                chunk, OMMIT_TPL = self._templatize_chunk(tpl_piece, txt_piece)
                                new_txt.append(chunk)
                    if not OMMIT_TPL:
                        txt = " ".join(new_txt)
                        #logging.debug(f"Found!! '{txt} using template '{tpl}")
                        CONTINUE_NEXT_TPL = False

        return txt

    def _templatize_chunk(self, tpl_piece, txt_piece, OMMIT_TPL=False):
        """
        Processes a chunk of text and replaces dynamic elements with wildcards.

        Parameters
        ----------
        tpl_piece : str
            Template piece containing placeholders.
        txt_piece : str
            Text piece to be matched against the template.
        OMMIT_TPL : bool, optional
            Flag to omit the template if matching fails. Default is False.

        Returns
        -------
        tuple
            Chunk with placeholders replaced, and the updated omission flag.        
        Examples:
        if special = §:
            f(p§f§x, prefix) = p{}f{}x
            f(p§f§x, pr123efix) = p{}f{}x
        """
        #logging.debug("Templatizing chunk")
        tpl_piece += " "
        txt_piece += " "
        I = len(tpl_piece)
        i = 0
        chunk = ""
        SKIP_CHAR = False
        FLAG_TO_UNSKIP = None

        for j in range(len(txt_piece)-1):
            if not OMMIT_TPL:
                if txt_piece[j] == FLAG_TO_UNSKIP:
                    SKIP_CHAR = False
                elif tpl_piece[i] == self.special:
                    FLAG_TO_UNSKIP = tpl_piece[i+1]
                    i += 1
                    # logging.debug(f"+= {self.wildcard}, flag_to_unskip = '{FLAG_TO_UNSKIP}'")
                    chunk += self.wildcard
                    SKIP_CHAR = True

                if not SKIP_CHAR:
                    if tpl_piece[i] != txt_piece[j]:
                        OMMIT_TPL=True
                        # logging.debug(f"skipped, not equal {i},{tpl_piece[i]} != {j}, {txt_piece[j]}")
                    else:
                        # logging.debug(f"+= {tpl_piece[i]}")
                        chunk += tpl_piece[i]
                        i += 1
                # else:
                    # logging.debug(f"skipped char '{txt_piece[j]}")

        if i != len(tpl_piece)-1:
            #logging.debug(f"STOP, {i} != {len(tpl_piece)-1} I reached the end of the txt_piece {len(tpl_piece)} and there are still tpl_piece {i}")
            OMMIT_TPL = True

        return chunk, OMMIT_TPL

    def save(self, path):
        """
        Saves the tokenizer object to a file.

        Parameters
        ----------
        path : str
            Path to save the tokenizer object.
        """        
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    def fit_on_traces(self, traces):
        pass

def _evaluate(serie1, serie2):
    l1 = len(set(serie1))
    l2 = len(set(serie2))
    res = {
        # 'original vocab': l1,
        'new vocab': l2,
        'compression': f'{(100 * (1- l2/l1 ) ):.2f}%'
    }
    return res

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
    # freq_vector = [ token_count[t] for t in token_vector ]
    return freq_vector    

def cluster_labels( M_TFV ):
    # Magic parameters! Needs explanation. 4 is to discard false clusters and have a minimum number of elements.
    X=np.vstack(M_TFV.to_numpy())
    dbscan = DBSCAN(eps=3, min_samples=2)

    # Fit the model to the data and obtain cluster labels
    labels = dbscan.fit_predict(X)
    return labels

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

def common_head(list_str):
    """
    For ['tcshdr_dcs_102' 'tcshdr_dcs_103' 'tcshdr_dcs_104'...]
    Return 'tcshdr_dcs_§'
    """
    i = 0
    common = ''
    try:
        while i < len(list_str[0]) and all([len(x) > i for x in list_str]) and all([x[i]==list_str[0][i] for x in list_str[1:]]):
            common += list_str[0][i]
            i += 1
    except Exception as e:
        print(f"An error occurred: {e}")

        print(f"i={i}")
        print(f"{list_str}")

        traceback.print_exc()
        raise e

    return common

def common_tail(list_str):
    """
    See common_head
    """
    i = 1
    common = ''
    while i < len(list_str[0]) and all([len(x) >= i for x in list_str]) and all([x[-i]==list_str[0][-i] for x in list_str[1:]]):
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





class VltLogColorizer(LogColorizer):
    def __init__(self, tokenizer_func=None) -> None:
        if not tokenizer_func:
            tokenizer_func = VltTokenizer()
        super().__init__(tokenizer_func)

import logging
import pickle
from collections import Counter
import pandas as pd
import numpy as np

def load_model(path):
    with open(path, 'rb') as f:
        myModel = pickle.load(f)
    myInstance = myModel.__class__()
    myInstance.__dict__.update(myModel.__dict__)
    return myInstance


# TODO: docstring
class Strangelog():
    """
    The analysis is precomputed in NORMAL traces splitted by subset, in this use case, subset=TPL_ID row. Each color in the whole vocabulary is labelled with:

    | Strangeness | Meaning | comment |
    |--------|---------|---------|
    | 0      | Boring  | Observed in 80% or more of the traces | 
    | 1      | Common  | Observed in 20% - 80% of the traces   | 
    | 2      | Rare    | Observed in 20% or less of the traces | 
    | 3      | Unexpected | Never seen in the subset | 
    """

    #TODO: add all relevant properties with docstrings
    strangeness_ = {}
    # meta_ ...

    # TODO: docstring
    def __init__(self, subset_col=None, tokenizer=None) -> None:
        # TODO: move to getter, setter
        self.tokenizer_ = tokenizer
        # TODO: move to getter, setter
        self.subset_col = subset_col

    # TODO: docstring
    def fit(self, meta, traces, warn=False):
        """
        Train the model with the given traces and metadata.
        
        Instructions
        meta must be filtered to normal cases, without errors or aborts. This way the model will learn the nornmal behavior and 
        will detect outliers based on deviations of this normality.
        """
        if isinstance(warn, bool) and warn:
            log = logging.warning
        else:
            log = lambda x: x

        if isinstance(meta, pd.DataFrame): 
            self.meta_ = meta.copy()  
        else:
            self.meta_ = None

        # Tokenize
        # ---------------
        log("Pass 1/3: Tokenize traces")
        traces['tokenized'] = traces['event'].apply(self.tokenizer_.tokenize)
        traces['color_id'] = traces['tokenized'].apply( lambda x: self.tokenizer_.vocab_dict_.get(x, 0) )
        log("Pass 1/3: Tokenize traces -- Done!")

        # Compute sequences
        # ---------------
        sequences = traces.groupby('trace_id').apply(lambda x: x['color_id'].to_list(), include_groups=False)

        # Ensure both DataFrames have the same index and are sorted consistently
        common_index = sorted(set(self.meta_.index) & set(sequences.index))
        self.meta_ = self.meta_.loc[common_index]
        sequences = sequences.loc[common_index]

        self.meta_['sequence'] = sequences

        # Vector encodings
        # ---------------
        log("Pass 2/3: Vector encodings")
        self.meta_['onehot']   = self.meta_['sequence'].apply(self.seq_to_onehot)
        self.meta_['countvec'] = self.meta_['sequence'].apply(self.seq_to_count_vectorizer)

        log("Pass 2/3: Vector encodings -- Done!")
        # Strangeness
        # ---------------
        log("Pass 3/3: Compute strangeness")
        self.strangeness_ = {}
        for subset_filter in self.meta_[self.subset_col].unique():
            df_subset = self.meta_[self.meta_[self.subset_col] == subset_filter]

            matrix = np.vstack(df_subset['onehot'].to_numpy())

            p100 = np.percentile(matrix, 100, axis=0).astype(int)
            p80  = np.percentile(matrix, 80, axis=0).astype(int)
            p20  = np.percentile(matrix, 20, axis=0).astype(int)

            c_boring     =  p20         * 0
            c_common     = (p80 - p20)  * 1
            c_rare       = (p100 - p80) * 2
            c_unexpected = (1 - p100)   * 3

            self.strangeness_[subset_filter] = [ max(c_boring[i], c_common[i], c_rare[i], c_unexpected[i]) for i in range(len(self.tokenizer_.colors_)) ]
        log("Pass 3/3: Compute strangeness -- Done!")

        log("------- Results -------")
        log('   [subset]            : {c_boring}, {c_common}, {c_rare}, {c_unexpected}')
        for key, val in self.strangeness_.items():
            c_boring = len([x for x in val if x == 0])
            c_common = len([x for x in val if x == 1])
            c_rare = len([x for x in val if x == 2])
            c_unexpected = len([x for x in val if x == 3])
            log(f'   {key:20}: {c_boring}, {c_common}, {c_rare}, {c_unexpected}')
        log("Finished!")


    # TODO: docstring
    def predict(self, subset_val, trace, event_col='event'):
        # From event, get tokenized then color_id
        color_id = trace[event_col].apply( lambda x: self.tokenizer_.vocab_dict_.get(self.tokenizer_.tokenize(x), 0) )

        # This generates a new column "strangeness" in trace 
        if subset_val in self.strangeness_.keys():
            # .apply( lambda x: self.tokenizer_.vocab_dict_.get(self.tokenizer_.tokenize(x), -1) )
            trace['strangeness'] = color_id.apply(lambda x: self.strangeness_[subset_val][x] )
        else:
            trace['strangeness'] = -1
        return trace

    # TODO: docstring
    def seq_to_onehot(self, indices):
        # dict_keys(['<unk>', 'log bob_ins started at {} underlined',  ...
        one_hot = np.zeros(len(self.tokenizer_.vocab_dict_.keys()))
        one_hot[indices] = 1
        return one_hot

    # TODO: docstring
    def seq_to_count_vectorizer(self, indices):
        c = Counter()
        c.update(indices)
        counts = [c[color_id] for color_id in range(len(self.tokenizer_.vocab_dict_.keys()))]
        counts_array = np.array(counts)
        return counts_array
    
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

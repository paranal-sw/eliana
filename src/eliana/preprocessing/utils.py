from collections import Counter
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .tokenizers import Numbers, UTCdate, load_tokenizer

# Set the warning mode to 'None' to suppress the warning SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def get_logvocab(df):
    """
    Returns a dictionary containing various statistics about the tokens and colors in the DataFrame.
    Recommended usage to have access to local variables:

    >>> locals().update( get_logvocab(df['tokenized event']) )
    >>> print(token_count, color_count, colors, tokens, df_tokens, df_colors)

    Args:
        df (pandas.Series): A pandas Series containing a tokenized column 

    Returns:
        dict: A dictionary containing the following keys:
            - 'token_count': A Counter object with the frequency of each token in the DataFrame.
            - 'color_count': A Counter object with the frequency of each color in the DataFrame.
            - 'colors': A list of unique colors in the DataFrame.
            - 'tokens': A list of unique tokens in the DataFrame.
            - 'df_tokens': A DataFrame with columns 'word' and 'frequency' representing the tokens and their frequencies, sorted by frequency.
            - 'df_colors': A DataFrame with columns 'word' and 'frequency' representing the colors and their frequencies, sorted by frequency.
    """
    res = {}
    res['token_count'] = Counter(' '.join(df).split(' '))
    res['color_count'] = Counter(df)
    res['colors'] = res['color_count'].keys()
    res['tokens'] = res['token_count'].keys()

    res['df_tokens']=pd.DataFrame({ 'token':          res['token_count'].keys(), 'frequency': res['token_count'].values() }).sort_values('frequency')
    res['df_colors']=pd.DataFrame({ 'tokenized event':res['color_count'].keys(), 'frequency': res['color_count'].values() }).sort_values('frequency')
    return res


def histogram_frequencies(df, label):
    """
    Create a histogram of the frequencies of a given label in a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the frequency data.
        label (str): The label for which the frequencies are calculated.

    Returns:
        None. The function only displays the histogram plot.
    """

    # Create logarithmically spaced bin edges
    log_bins = np.logspace(np.log10(min(df['frequency'])), np.log10(max(df['frequency'])), 20)  # 20 bins

    # Create the histogram with log bins
    plt.hist(df['frequency'], bins=log_bins, edgecolor='k')

    plt.grid(True)

    plt.ylabel(f"# of {label}s at given frequency")
    plt.xlabel(f"frequency (occurrences of {label} inside the corpus)")
    plt.xscale('log')

    # Define a custom tick formatter function
    def format_ticks(x, pos):
        if x < 1e4:
            return f"{int(x)}"
        else:
            return f"{x:.0e}"

    # Apply the custom tick formatter to the X-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
 
    plt.title(f"{label} Frequency Histogram")
    plt.show()


def _measure_growth(dframe, vocab_type, plot=False, title=None, retplot=False):
    """
    Calculates the growth of a vocabulary (either token or color) over a given DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the text data.
        vocab_type (str, optional): The type of vocabulary to measure the growth of (either 'Token' or 'Color'). Defaults to 'Token'.
        plot (bool, optional): Whether to plot the vocabulary growth or not. Defaults to False.

    Returns:
        float: The slope of the vocabulary growth, multiplied by 1,000,000.
    """
    # Randomize to avoid artifacts
    df=dframe.sample(frac=1).reset_index(drop=True)
    chunk = int(len(df)/500)+2
    steps = int(len(df)/chunk)

    c = Counter()
    x, y = [], []

    for i in range(0, steps):
        if vocab_type == 'Token':
            c.update(' '.join(df[i*chunk:(i+1)*chunk]).split(' '))
        elif vocab_type == 'Color':
            c.update( df[i*chunk:(i+1)*chunk] )
        else:
            raise ValueError('measure_growth(df, vocab_type) vocab_type is required')

        # Collect size of trace examples
        x.append( i*chunk )

        # Collect vocabulary growth 
        y.append( len(c)) 

    # Calculate the gradient of y with respect to x
    gradient = np.gradient(np.array(y), np.array(x))

    # Compute the vocabulary_slope
    vocabulary_slope = np.median(gradient)

    if plot:
        plt.scatter(x, y, marker='.', color='b')

        # Calculate the central values
        central_x = x[len(x) // 2]
        central_y = y[len(y) // 2]

        # Plot the vocabulary_slope line tangent to central_x, central_y
        b = central_y - vocabulary_slope * central_x
        x2 = np.linspace(min(x), max(x), 100)
        y2 = vocabulary_slope * x2 + b

        str_frac = "$\\frac{words}{(M) events}$"
        plt.plot(x2, y2, label=f'{vocab_type} $\mu$ slope={vocabulary_slope*1000000:.0f} {str_frac}', color='red')

        plt.xlabel(f'# of events to make the {vocab_type}')
        plt.ylabel(f'# of unique {vocab_type}s')
        plt.ylim(bottom=0)
        if title:
            plt.title(title)
        else:
            plt.title(f'Unique {vocab_type} Growth')
        plt.grid(True)
        plt.legend()
        if retplot:
            return plt
        else:   
            plt.show()

    return vocabulary_slope*1000000

def measure_token_growth(df, plot=False, title=None, retplot=False):
    """
    Calculates the growth of a vocabulary of tokens over a given DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the text data.
        plot (bool, optional): Whether to plot the vocabulary growth or not. Defaults to False.

    Returns:
        float: The slope of the vocabulary growth, multiplied by 1,000,000.
    """
    return _measure_growth(df, vocab_type='Token', plot=plot, title=title, retplot=retplot)

def measure_color_growth(df, plot=False, title=None, retplot=False):
    """
    Calculates the growth of a vocabulary of colors over a given DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the text data.
        plot (bool, optional): Whether to plot the vocabulary growth or not. Defaults to False.

    Returns:
        float: The slope of the vocabulary growth, multiplied by 1,000,000.
    """
    return _measure_growth(df, vocab_type='Color', plot=plot, title=title, retplot=retplot)


def generate_events(n_events, n_templates, seed=None, max_tokens_per_tpl=20, p_fixed_token=0.7, p_numeric_token=0.25, p_composed_token=0.05):
    """
    Generate a specified number of events based on a given number of templates and other parameters.

    Args:
        n_events (int): The number of events to generate.
        n_templates (int): The number of templates to generate for each type.
        seed (int, optional): The seed value for the random number generator. Defaults to None.
        max_tokens_per_tpl (int, optional): The maximum number of tokens per template. Defaults to 20.
        p_fixed_token (float, optional): The probability of selecting a fixed token template. Defaults to 0.7.
        p_numeric_token (float, optional): The probability of selecting a numeric token template. Defaults to 0.25.
        p_composed_token (float, optional): The probability of selecting a composed token template. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the generated templates for each type.
        list: A list of the generated events (traces).
        list: A list of the tokenized versions of the events.
    """
    if seed:
        random.seed(seed)

    T = 'a b c d e f g h i j k l m n o p q r s t u v w x z'.split()
    def symbol(n=3):
        return ''.join([random.choice(T) for _ in range(n)])

    # Generate a fixed list of templates
    templates = {}

    tpls = []
    for _ in range(int(n_templates*p_fixed_token)):
        N = random.randint(1, max_tokens_per_tpl)
        tpl = ' '.join([ symbol(3) for n in range(N) ])
        tpls.append(tpl)
    templates['fixed_token'] = list(set(tpls))

    tpls = []
    for _ in range(int(n_templates*p_numeric_token)):
        N = random.randint(1, max_tokens_per_tpl)
        tpl = '{} ' + ' '.join([ symbol(3) for n in range(N-1) ])
        tpls.append(tpl)
    templates['numeric_token']=list(set(tpls))

    tpls = []
    for _ in range(int(n_templates*p_composed_token)):
        N = random.randint(1, max_tokens_per_tpl)
        tpl = 'aa XX{} ' + ' '.join([ symbol(3) for n in range(N-1) ])
        tpls.append(tpl)
    templates['composed_token']=list(set(tpls))

 
    traces = []
    tokenized = []
    for _ in range(n_events):
        p = random.random()
        if p < p_fixed_token:
            S = templates['fixed_token']
        elif p < p_fixed_token + p_numeric_token:
            S = templates['numeric_token']
        else:
            S = templates['composed_token']
        tpl = random.choice(S)
        traces.append(tpl.replace('{}', str(random.randint(0, 100000))))
        tokenized.append(tpl)

    return templates, traces, tokenized
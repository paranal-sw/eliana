import re
import pickle
from types import SimpleNamespace

def load_tokenizer(path):
    with open(path, 'rb') as f:
        myColor2 = pickle.load(f)
    newColor = myColor2.__class__()
    newColor.__dict__.update(myColor2.__dict__)
    return newColor


import re
import pickle
from types import SimpleNamespace

class AbstractTokenizer():
    """
    Main methods for tokenizers.
    """
    
    options = SimpleNamespace()

    _reg_spaces = re.compile(r"(\ {2,}|\t+)")


    def __init__(self, **kwargs) -> None:
        self.options = SimpleNamespace(
            remove_extra_spaces = False, 
            to_lowercase = False,
            strip = False
        )    
        for k in kwargs.keys():
            if k in self.options.__dict__:
                setattr(self.options, k, kwargs[k])

    def normalize(self, txt):
        """
        Applies option normalization to the input text.
        """
        if self.options.to_lowercase:
            txt = txt.lower()
        if self.options.remove_extra_spaces:
            txt = self._reg_spaces.sub(' ', txt)
        if self.options.strip:
            txt = txt.strip()
        return txt
    
    def tokenize(self, txt):
        return self.normalize(txt)

    def help(self):
        """
        Provides help information about the tokenizer class, including its inheritance and tokenization order.

        Returns
        -------
        str
            Help information.
        """        
        class_name = f'Tokenizer class "{self.__class__.__name__}"  '
        help_text = [class_name, '=' * (len(class_name) - 2)]
        help_text.append('Inherits from: {}'.format(', '.join([
             x.__name__ for x in self.__class__.__mro__ 
             if x not in [object, self.__class__] 
             and x.__name__ not in "AbstractTokenizer"
             ])))
        help_text.append(f"Options: {self.options}")
        help_text += ["", "Tokenization is done in the following order"]
        
        for parent_class in [ 
             x for x in self.__class__.__mro__ 
             if x not in [object, self.__class__] 
             and x.__name__ not in "AbstractTokenizer"
             ]:
            if parent_class.__doc__:
                help_text.append(" ")
                help_text.append(f'{parent_class.__name__}: ' + parent_class.__doc__)
        if self.__doc__:
            help_text.append(" ")
            help_text.append(f'{self.__class__.__name__}: ' + self.__doc__)
        return "\n".join([line.strip() for line in help_text])    

class RegExpTokenizer(AbstractTokenizer):
    """    A tokenizer that uses regular expressions to tokenize text. Inherits from AbstractTokenizer. """
    my_regexps = []

    def __init__(self, **kwargs) -> None:
        """
        Initializes the RegExpTokenizer with optional keyword arguments.
        
        kwargs : dict
            Optional keyword arguments to initialize the tokenizer.
        """
        AbstractTokenizer.__init__(self, **kwargs)

        self.regexps = []

        # for parent_class in [x for x in self.__class__.__mro__ if x not in [object, self.__class__]]:
        for parent_class in [x for x in self.__class__.__mro__ if x not in [object]]:
            for regex in getattr(parent_class, 'my_regexps', []):
                if regex not in self.regexps:
                    self.regexps.append(regex)

        self._compiled_regexps_list = None
        self.add_regexp( r'(\{\})+', '{}')


    def add_regexp(self, pattern, replacement):
        """
        Adds a new regular expression to the tokenizer. Regexps are used as in
        re.sub(pattern, replacement)

        Parameters
        ----------
        pattern : str
            Regular expression pattern to search for.
        replacement : str
            Replacement string for the pattern.
        """        
        self.regexps.append((pattern, replacement))
        self._compiled_regexps_list = []
        
    def tokenize(self, txt):
        self._compiled_regexps

        # REGEXP
        # ------
        for compiled, replacement in self._compiled_regexps:
            txt = compiled.sub(replacement, txt)

        # Normalize
        # ---------
        txt = self.normalize(txt)

        return txt

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

    @property
    def _compiled_regexps(self):
        """
        Compiles the regular expressions in self.regexps into a list of compiled patterns and replacements.
        """        
        if not self._compiled_regexps_list:
            self._compiled_regexps_list = [(re.compile(pattern), replacement) for pattern, replacement in self.regexps]
        return self._compiled_regexps_list


class BaseTokenizer:
    """
    The BaseTokenizer class provides basic tokenization functionality based on regular expressions.
    It supports adding 1) regular expressions and 2) templates, and offers methods for tokenizing
    strings and extracting parameters from tokenized strings.
    """

    # Used for configuration
    options = SimpleNamespace()
    regexps = []

    _stats = None
    _templates = None
    _reg_spaces = re.compile(r"(\ {2,}|\t+)")
    _reg_double_wildcards = re.compile(r"{}{}")
    _template_prefixes = []
    _word = None
    _compiled_regexps = None

    def __init__(self, word=""):
        """
        Initialize the BaseTokenizer with optional initial word.

        Inherits regular expressions from parent classes and sets initial statistics.

        Parameters
        ----------
        word : str, optional
            Initial word to be tokenized.
        """
        for parent_class in [x for x in self.__class__.__mro__ if x not in [object, self.__class__]]:
            for regex in parent_class.regexps:
                if regex not in self.regexps:
                    self.regexps.append(regex)

        if word:
            self._word = word

        self._stats = {
            'events': 0,
            'traces': 0,
            'uniques': 0,
            'compression': 0,
            'templates': None,
            'regexp': None,
            'growth_rate': -1
        }
        self._templates = []
        self._compiled_regexps = None
        self._template_prefixes = []

        self.options = SimpleNamespace(
            spaces=True, 
            lowercase=False
        )

    @property
    def stats(self):
        """
        Returns a dict with the current statistics of the tokenizer, including the count of 
        templates and regular expressions.
        """
        self._stats['templates'] = len(self.templates)
        self._stats['regexp'] = len(self.regexps)
        return self._stats

    @property
    def templates(self):
        """
        A list containing the templates. Templates are predefined patterns that help in transforming
        and tokenizing text. They contain placeholders represented by the character '§' which are
        replaced by '{}' during tokenization. This allows for consistent handling of similar text
        patterns.
        
        For example, given the following string and template:

        a = "Getting header from VLTI at exposure start (TCSHDR_DCS_203)"
        b = "Unchanged string"
        TPL = 'Getting header § at exposure start (TCSHDR_DCS_§)'

        This fake function `replace_template` replicates how we transform the string based on the template.

        > replace_template(TPL, a)
        'Getting header {} at exposure start (TCSHDR_DCS_{})'

        > replace_template(TPL, b)
        'Unchanged string' 

        The first string matches the template and gets transformed, while the second string does not
        match the template and remains unchanged. This method is far quicker than using regexps.
        """
        return self._templates
    
    @templates.setter
    def templates(self, val):
        self._templates = val
        self._compile_templates()

    def add_regexp(self, pattern, replacement):
        """
        Adds a new regular expression to the tokenizer. Regexps are used as in
        re.sub(pattern, replacement)

        Parameters
        ----------
        pattern : str
            Regular expression pattern to search for.
        replacement : str
            Replacement string for the pattern.
        """        
        self.regexps.append((pattern, replacement))
        self._compile_regexps()

    def help(self):
        """
        Provides help information about the tokenizer class, including its inheritance and tokenization order.

        Returns
        -------
        str
            Help information.
        """        
        class_name = f'Tokenizer class "{self.__class__.__name__}"  '
        help_text = [class_name, '=' * (len(class_name) - 2)]
        help_text.append('Inherits from: {}'.format(', '.join(
            [x.__name__ for x in self.__class__.__mro__ if x not in [object, self.__class__, BaseTokenizer]])))
        help_text += ['Tokenization is done in the following order:']
        
        for parent_class in [x for x in self.__class__.__mro__ if x not in [object, self.__class__, BaseTokenizer]]:
            if parent_class.__doc__:
                help_text.append(" ")
                help_text.append(f'{parent_class.__name__}: ' + parent_class.__doc__)
        if self.__doc__:
            help_text.append(" ")
            help_text.append(f'{self.__class__.__name__}: ' + self.__doc__)
        return "\n".join([line.strip() for line in help_text])

    def tokenize(self, text):
        """
        Tokenizes the input text using the configured regular expressions and templates. 
        The symbol {} are used for wildcard.

        The method applies the following steps:
        1. Adds a space at the beginning and end of the text to handle edge cases.
        2. Applies options for lowercase conversion and space normalization.
        3. Applies regular expressions to transform the text.
        4. Replaces double wildcards with a single wildcard.
        5. Applies templates to further transform the text.
        6. Removes leading and trailing spaces.        

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        str
            Tokenized text.
        """        
        if not self._compiled_regexps:
            self._compile_regexps()

        # Add a space at the beginning of the string
        text = f' {text} '

        # OPTIONS
        # ----
        if self.options.lowercase:
            text = text.lower()

        if self.options.spaces:
            text = self._reg_spaces.sub(' ', text)

        # REGEXP
        # ------
        for compiled, replacement in self._compiled_regexps:
            text = compiled.sub(replacement, text)

        text = self._reg_double_wildcards.sub('{}', text)

        if self.options.spaces:
            text = self._reg_spaces.sub(' ', text)

        # Leading spaces
        text = text.strip()

        # TEMPLATES
        # ---------
        for idx in range(len(self.templates)):
            text = self._replace_template(idx, text)

        if self.options.spaces:
            text = self._reg_spaces.sub(' ', text)

        return text

    def extract_params(self, original, pattern=None):
        """
        Extracts parameters from the original string using tokenization

        !TODO: this method is very bugged, needs refactoring.

        Parameters
        ----------
        original : str
            Original string to extract parameters from.
        pattern : str, optional
            Pattern string to use for extraction. If not provided, it is generated from self.tokenize

        Example:
        
        original="LOG bob_ins   Started at 2019-04-10T10:14:14 (underlined)"
        pattern="LOG bob_{}   Started at {} (underlined)"

        extract_params(original, pattern)
        ["ins", "2019-04-10T10:14:14"]

        Returns
        -------
        list
            List of extracted parameters.
        """
        if not pattern:
            pattern = self.tokenize(original)
        
        pattern = f' {pattern} '
        original = f' {self._reg_spaces.sub(" ", original)} '

        chunks = pattern.split('{}')

        if len(chunks) <= 1:
            return []
        else:
            params = []
            for a, b in zip(chunks[:-1], chunks[1:]):
                param = original.split(a)[1].split(b)[0]
                params.append(param)
        return params

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

    def _compile_regexps(self):
        """
        Compiles the regular expressions in self.regexps into a list of compiled patterns and replacements.
        """        
        self._compiled_regexps = [(re.compile(pattern), replacement) for pattern, replacement in self.regexps]

    def _compile_templates(self):
        """
        Compiles the templates in self.templates by extracting and storing their prefixes,
        'Getting header § at exposre start (TCSHDR_DCS_§)' -> 'Getting header '
        """        
        self._template_prefixes = [template.split('§')[0] for template in self.templates]

    def _replace_template(self, idx, text):
        """
        Replaces template placeholders in the input text.

        Example:
        
        a = "Getting header from VLTI at exposre start (TCSHDR_DCS_203)"
        b = "Unchanged string"
        TP = 'Getting header § at exposre start (TCSHDR_DCS_§)'

        > replace_template(TP, a)
        Getting header {} at exposre start (TCSHDR_DCS_{})

        > replace_template(TP, b)
        Unchanged string
        """
        template = self.templates[idx]

        # Optimization 1: trivial
        if len(template) > len(text):
            return text
        
        # Optimization 2: prefix lengths
        prefix = self._template_prefixes[idx]
        if len(prefix) > len(text) or text[:len(prefix)] != prefix:
            return text
        
        # Proceed with template replacements
        tokens = []
        remaining_text = text
        keep_going = True
        parts = template.split('§')
        parts.reverse()

        while parts and keep_going:
            chunk = parts.pop()
            try:
                i = remaining_text.index(chunk)
                remaining_text = remaining_text[len(chunk) + i:]
                tokens.append(chunk)
            except:
                keep_going = False

        if keep_going:
            return '{}'.join(tokens)
        else:
            return text



class Numbers(RegExpTokenizer):
    """Transform numbers using {} as token, but ignore numbers that are parts of a word
    1 -> {}
    -10.54 -> {}
    9.1e-2.1 -> {}
    There are 2 telescopes: UT1, UT2 -> There are {} telescopes: UT1, UT2
    2Good2be_True -> 2Good2be_True
    """
    my_regexps = [
        (r'\b-?\d+(\.\d+)?([eE][-+]?\d+(\.\d+)?)?\b', '{}'),
        ('-{}', '{}')
    ]

class UTCdate(RegExpTokenizer):
    """Transform UTC dates using {} as token
    2022-10-01T00:43:01.123 -> {}
    Started at 2019-04-01T22:29:07 (underlined) -> Started at {} (underlined)
    """
    my_regexps = [
        (r"\d{4}-\d{2}-\d{2}[ tT]\d{2}:\d{2}:\d{2}(\.\d{0,3})?", r"{}") # UTCdate
    ]

class Punctuation(RegExpTokenizer):
    """Remove all punctuation
    Original: Hi! I'm counting 1,2, 3 ... and so on.
    Tokenized: Hi I m counting 1 2 3 and so on
    """
    my_regexps = [
        (r"[\"'!,;:\+\*\$<>\.\-|/\\=\[\]\()#]", r" ")
    ]

class VltTokenizer(UTCdate, Numbers, Punctuation):
    """Domain specific transformation for Paranal VLT software logs using {} as token
    Original : wat2tcs lt3aga w2fors (bobWish_105797) ITERATION=10 [bob_234] lt4aag cmd77 2022-10-01T00:43:01.123 
    Tokenized: wat{}tcs lt{}aga w2fors ( bobwish_{} iteration={} [ bob_{} lt{}aag cmd{} {}

    Original : ... "B02_HD95578_MED_SPLIT" OBS.OBSERVER "UNKNOWN" OBS.PI-COI.ID "70033" OBS.PI-COI.NAME "UNKNOWN"
    Tokenized: _setup_parameters_ommited_
    """
    def __init__(self):
        super().__init__()
        self.options.remove_extra_spaces=True
        self.options.to_lowercase=True
        self.options.strip = True

    my_regexps = [
       (r"([lw]a{0,1}t)[0-9]([a-z]+)", r"\1{}\2"), #WS and LCU 
       (r"(\W)cmd\d+", r"\1cmd{}"), # CSS commands
    #    (r"\s*?bob\S+\s+", r" bob{} "), # bob names
       (r"(\s*?[a-z][a-z0-9]{2,})_[0-9]{3,}(\s*)", r"\1_{}\2"), # bob names
       (r"\.\.\.(\s+\S+){5}.*", "_setup_parameters_ommited_"),
    ]

class Vlt2Tokenizer(BaseTokenizer):
    """    Old version of tokenizers using Base Tokenizer
    2022-10-01T00:43:01.123 -> {}
    Hi! I'm counting 1,2, 3 ... and so on.
    Original : wat2tcs (bobWish_105797) ITERATION=10 [bob_234] lt4aag cmd77 2022-10-01T00:43:01.123 
    Tokenized: wat{}tcs bob{} iteration {} bob{} lt{}aag cmd{} {}
    Original : ... "B02_HD95578_MED_SPLIT" OBS.OBSERVER "UNKNOWN" OBS.PI-COI.ID "70033" OBS.PI-COI.NAME "UNKNOWN"
    Tokenized: _setup_parameters_ommited_

    """
    regexps = [
        # UTCDATE
        (r"\d{4}-\d{2}-\d{2}[ tT]\d{2}:\d{2}:\d{2}(\.\d{0,3})?", r"{}"),
        # Numbers
        (r'\b-?\d+(\.\d+)?([eE][-+]?\d+(\.\d+)?)?\b', '{}'),
        ('-{}', '{}'),
        # Punctuation
        (r"[\"'!,;:\+\*\$<>\.\-|/\\=\[\]\()#]", r" "),
        # VLT Specific
       (r"([lw]a{0,1}t)[0-9]([a-z]+)", r"\1{}\2"), #WS and LCU 
       (r"(\W)cmd\d+", r"\1cmd{}"), # CSS commands
       (r"\s*?bob\S+\s+", r" bob{} "), # bob names
       (r"\.\.\.(\s+\S+){5}.*", "_setup_parameters_ommited_")
    ]

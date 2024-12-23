"""
Log Parsers Utilities

"""
import logging
import re
import pandas as pd
        
class Log2Table:
    """
    Parse a list of events following a criteria based in state machines to fill a table with complex behavior. 
    """
    def __init__(self, **kwargs):
        """
        Constructor
        ----------
        timestamp : str, optional. Column name for timestamp.
        timeout : str
        truncated_rows : bool, allow to continue parsing with missing data
        """
        self._conf = kwargs
        self._rows = {}
        self._start = {}
        # Similar to states
        self._whens = []
        if 'timeout' in self._conf:
            pass

    def start_when(self, **kwargs):
        """
        Declare initialization of a row

        Parameters
        ----------
        receive : str or function.
        fields_id : list, optional. Allows different rows from parallel sources
        cols : list, optional. Initial list of columns
        if_has_elapsed : int, optional. Attempts row creation if_has_elapsed=X seconds
        has_elapsed : int. Start new row when has_elapsed at least X seconds since previous log

        """
        self._start = kwargs
        if 'fields_id' in self._start:
            self._conf['fields_id'] = self._start['fields_id']
    
    def when(self, **kwargs):
        """
        Template to fill a column in a row based on its content

        Parameters
        ----------
        receive : str or function
        set_cols : list
        offset: float, optional. Add offset to timestamp.
        overwrite: bool, default True. Allow overwrite previous recorded values.
        end_row : bool

        Use fields_id to have parallel rows from different sources

        """
        if 'overwrite' not in kwargs:
            kwargs['overwrite'] = True
        self._whens.append( kwargs )

        #TODO: Check is dict
        # if 'set_cols' in when.keys()

        #TODO: Check is bool
        # if 'end_row' in when.keys()

        #TODO: If time_resolution is used, check self._conf['timestamp']


    def parse(self, user_df):
        """
        Iterate DataFrame or list and create rows previously defined

        Parameters
        ----------
        df : DataFrame or list of dicts

        Returns
        -------
        DataFrame
        """

        df = user_df.copy()

        logging.info(f"Start parsing with query of length = {len(df)}")
        #TODO: Check integrity of df


        # If timestamp is used, sort by timestamp
        if 'timestamp' in self._conf:

            # Check time_resolution in whens
            # Trick: repeat with delta=time_resolution to lower resolution events
            def add_delta_T(log):
                delta = 0
                for when in self._whens:
                    if 'offset' in when:
                        log['__allfields'] = ' '.join( [ f'{k}: {v}' for k, v in log.items() ] )
                        in_col =  when['in_col'] if 'in_col' in when else '__allfields'
                        if self._receive_event(log, when['receive'], in_col=in_col):
                            delta = when['offset']*1000000000
                return delta
            df['__add_delta_T'] = df.apply( add_delta_T, axis=1)
            df[self._conf['timestamp']] = df[self._conf['timestamp']] + df['__add_delta_T'].astype('timedelta64[ns]')

            # If timestamp is used, sort by timestamp
            df.sort_values(by=self._conf['timestamp'], inplace=True)
            # print(df[ df['__add_delta_T'] > 0 ][['@timestamp', 'keywname', '__add_delta_T']][:50])
            self.df = df

        self._rows = {}
        self._rows_list = []
        self._previous_log = False
        since_previous_log = False

        ts = False
        for i, log in df.iterrows():

            if 'timestamp' in self._conf:
                ts = log[self._conf['timestamp']]

            # Concatenate elements in a simple str
            log['__allfields'] = ' '.join( [ f'{k}: {v}' for k, v in log.items() ] )
            logging.debug(f'- Analyzing row {i} = {log.logtext}')

            # Iterate over existing machines
            remove_machines = []

            # TODO: has_elapsed needs to be thoroughful tested!!!
            if 'has_elapsed' in self._start:
                # TODO: in_col
                # has_passed enough time since previous, force new row creation
                if since_previous_log and self._has_passed_enough_time(log):
                    logging.debug(f'-- END Row: {name}')

                    if 'timestamp' in self._conf:
                        self._rows[name]['END'] = log[self._conf['timestamp']]
                    self._rows_list.append(self._rows[name])
                    # Remove previous machine
                    del(self._rows[name])

                    logging.debug('-- enoug time has passed, trigger new row')
                    self._create_row(log)

                elif since_previous_log and not self._has_passed_enough_time(log):
                    pass
                elif self._receive_event(log, self._start['receive']):
                    self._create_row(log)

            # Check starts by direct condition
            elif self._receive_event(log, self._start['receive']):
                self._create_row(log)


            for name, machine in self._rows.items():
                logging.debug(f'-- Checking machine {name}')
                
                for when in self._whens:

                    in_col =  when['in_col'] if 'in_col' in when else '__allfields'
                    if self._receive_event(log, when['receive'], in_col=in_col):
                        # logging.warning(f'{name} - Found {when["receive"]}')

                        if 'set_cols' in when:
                            for colname, colvalue in when['set_cols'].items():
                                if colname not in self._rows[name]:
                                    raise ValueError(f'the column "{colname}" is not present in declared cols')
                                value = self._parseLogFunc(log, colvalue)
                                # Do not allow empty!
                                if value != "":
                                    if when['overwrite'] or self._rows[name][colname]==None:
                                        self._rows[name][colname] = value
                                        # logging.warning(f"{name} -- colname={colname} = {value}")
                                    # else:
                                    #     logging.warning(f"{name} -- colname={colname} X : {when['overwrite']} or {self._rows[name][colname]}==None")
                                # else:
                                    # logging.warning(f"EMPTY!! colname={colname} log='{log.to_list()}', value='{value}")
                                    # pass

                        if 'end_row' in when and when['end_row']:
                            remove_machines.append(name)
                            if 'timestamp' in self._conf:
                                self._rows[name]['END'] = log[self._conf['timestamp']]
                            self._rows_list.append(self._rows[name])     
                            # logging.warning(f'-- END Row: {name}')


            for name in remove_machines:
                del(self._rows[name])

            # Remember me, but forget my fate (Dido & Eneas)
            self._previous_log = log
            since_previous_log = True

                
        logging.warning(len(self._rows_list))
        self._pandas = pd.DataFrame(self._rows_list)

        # display(rows_list)
        if len(self._pandas) and 'timestamp' in self._conf:
            self._pandas['START'] = pd.to_datetime(self._pandas['START'])
            self._pandas['END'] = pd.to_datetime(self._pandas['END'])
            self._pandas['SECONDS'] = (self._pandas['END'] - self._pandas['START']).astype('timedelta64[s]').astype('int')
            # self._pandas['SECONDS'] = (self._pandas['END'] - self._pandas['START']).astype('int')

        logging.info(f"Parsing finished, extracted {len(self._pandas)} rows")
        return self._pandas


    def _has_passed_enough_time(self, log):
        now = pd.to_datetime(log[self._conf['timestamp']])
        begin = pd.to_datetime(self._previous_log[self._conf['timestamp']])
        elapsed = pd.to_timedelta(now-begin).total_seconds()
        logging.debug(f'{now} -- {begin} -- {elapsed}')
        return elapsed >= self._start['has_elapsed']


    def _create_row(self, log):
        name = self._name(log)

        # Check if has elapsed enough time
        if 'if_has_elapsed' in self._start and \
        name in self._rows and \
        'START' in self._rows[name]:
            recently_created = self._start['if_has_elapsed']
            now = pd.to_datetime(log[self._conf['timestamp']])
            begin = pd.to_datetime(self._rows[name]['START'])
            elapsed = pd.to_timedelta(now-begin).total_seconds()
            logging.debug(f'{now} -- {begin} -- {elapsed}')
            if elapsed < recently_created:
                logging.debug(f'{name} was created recently, skipping row creation')
                return

        if name in self._rows:
            if 'truncated_rows' in self._conf:
                logging.debug(f'-- END Row but TRUNCATED: {name}')
                if 'timestamp' in self._conf:
                    self._rows[name]['END'] = log[self._conf['timestamp']]
                self._rows_list.append(self._rows[name])     

            else:
                raise ValueError(f'There is another unfinished row: {name} = {self._rows[name]} ----' + \
                                f' attempting to use {log["__allfields"]}')

        logging.debug(f'-- New Row: {name}')
        self._rows[name] = {}
        if 'timestamp' in self._conf:
            self._rows[name]['START'] = log[self._conf['timestamp']]
            self._rows[name]['END'] = -1


        if 'timeout' in self._conf:
            self._rows[name]['TIMEOUT'] = False

        # Check the initial columns of the row
        if 'cols' in self._start:
            for colname, colvalue in self._start['cols'].items():
                self._rows[name][colname] = self._parseLogFunc(log, colvalue)
        # display(self._rows[name])


    def _parseLogFunc(self, log, func):
        if func==None or type(func) in [str, int, bool, float]:
            return func
        else:
            return func(log)


    def _name(self, log):
        if 'fields_id' not in self._conf:
            return 'default'
        else:
            return '_'.join([ log[x] for x in self._conf['fields_id']])

    def _receive_event(self, real, expected, in_col='__allfields'):
        """Check if s against the template
        """        
        if type(expected) == str:
            return expected in real[in_col]
        else:
            result = expected(real[in_col])
            if result:
                logging.debug(f'{expected} found in {real[in_col]}')
            return result
    


    
# Helper functions
# Example: 
# after_a = lp.get_word_after('b')
# after_a('a b c')
# c
def get_word_after(txt, in_col='__allfields', delimiter=' '):
    # nonalpha = re.compile('\W')
    def func(x):
        if hasattr(x, 'keys') and in_col in x:
            return func(x[in_col])             
        elif isinstance(x, str):
            # multispace = re.compile(' +?')
            x = re.sub(r'\s+', r' ', x)
            items = x.split(delimiter)
            # logging.warning(f"'{txt}' in {items} ?")
            if txt in items:
                # logging.warning(f"Yes <3 {items.index(txt)+1}={items[items.index(txt)+1]}")
                return items[items.index(txt)+1]

    return func

def get_first_word(in_col='__allfields', delimiter=' '):
    nonalpha = re.compile('\W')
    def func(x):
        if hasattr(x, 'keys') and in_col in x:
            return func(x[in_col])       
        else: 
            items = [re.sub(nonalpha, '', a) for a in x.split(delimiter)]
            try:
                # Remove symbols
                return items[0]
            except:
                return ''
    return func

# Returns log[col] for a given column=col
def get_col_value(col):
    def func(x):
        return x[col]
    return func


def any_of(myList, in_col='__allfields'):
    def func(x):
        return any( [ item in x[in_col] for item in myList ] )
    return func

class txt:
    """
    Generic text with boolean
    
    Usage:
        condition = _(' (red)') & ~_('ACK ABORT')
        print(condition("yo soy (red)!"))  # > True
        print(condition("ACK ABORT (red)!"))  # > False

        condition_and = _(' (red)') & _('green')
        print(condition_and("yo soy (red) green!"))  # > True
        print(condition_and("yo soy green!"))  # > False

        condition_or = _(' (red)') | _('green')
        print(condition_or("yo soy (red)!"))  # > True
        print(condition_or("yo soy green!"))  # > True
        print(condition_or("yo soy blue!"))  # > False
    """
    def __init__(self, substring, in_col='__allfields'):
        self.substring = substring
        self.negate = False
        self.and_condition = None
        self.or_condition = None
        self.in_col = in_col

    def __call__(self, text):
        # Evaluate the condition based on the substring and negation
        result = self.substring in text
        if self.negate:
            result = not result

        # Apply and_condition if it exists
        if self.and_condition:
            result = result and self.and_condition(text)

        # Apply or_condition if it exists
        if self.or_condition:
            result = result or self.or_condition(text)

        return result

    def __and__(self, other):
        if isinstance(other, txt):
            self.and_condition = other
            return self
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, txt):
            self.or_condition = other
            return self
        return NotImplemented

    def __invert__(self):
        # Create a new Text object with negation
        inverted = txt(self.substring)
        inverted.negate = not self.negate
        return inverted
    
# Convenient alias
_ = txt
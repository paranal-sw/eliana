import os
import base64
import hashlib
import logging
import pandas as pd

# Suppress the SettingWithCopyWarning (use with caution)
pd.options.mode.chained_assignment = None

def _format_to_datetime(timestamp):
    """
    Convert a timestamp to a pandas datetime object.

    Parameters
    ----------
    timestamp : pd.Timestamp, int, float, str
        The timestamp to convert. Can be in pd.Timestamp, milliseconds or a generic datetime format.

    Returns
    -------
    datetime
        A pandas datetime object without timezone information.

    Raises
    ------
    TypeError
        If the timestamp is not a pd.Timestamp, int, float, or str.
    """
    if isinstance(timestamp, (int, float)):
        return pd.to_datetime(timestamp, unit='ms', utc=True).tz_localize(None)
    elif isinstance(timestamp, str):
        return pd.to_datetime(timestamp, utc=True).tz_localize(None)
    elif isinstance(timestamp, pd.Timestamp):
        return timestamp
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}. Expected int, float, or str.")

class Config:
    """
    Configuration class to store key-value pairs.

    Parameters
    ----------
    **kwargs : dict
        Key-value pairs to be stored as attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        """
        Convert the configuration attributes to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return self.__dict__

class ElianaDataset:
    """
    High level class for Eliana Dataset Manipulation, oriented to pandas DataFrames.
    """

    def __init__(self, **kwargs):
        """
        Common methods for Eliana Dataset Manipulation.

        The most interesting functionality of the class is that it generates a unique cache that speeds up queries or file manipulations.

        Parameters
        ----------
        config : dict
            A configuration dictionary containing key-value pairs for the dataset setup.

        Attributes
        ----------
        config : Config
            Configuration settings including required, optional, and default values.
        """
        dataset_name = kwargs.get('dataset_name', self.__class__.__name__)

        default_config = {
            'cols': None,
            'dataset_dir': 'data/raw',
            'dataset_name': dataset_name,
            'event_cols': None,
            'meta_filename': '_index',
            # 'processed_dir': 'data/processed',
            'trace_prefix': dataset_name,
            'trace_suffix': '',
            'use_cache': True
        }
        default_config.update(kwargs)

        self.config = Config(**default_config)
        self._meta = None
        self._df_traces = None
        self._trace_hash = False
        self._dict_for_trace_filenames = {}

    @property
    def meta(self):
        """
        Contains the metadata of the dataset.
        
        If the metadata is not in memory, it tries to load from the cache file at `self.meta_filename`.
        If the cache file doesn't exist, it tries to generate and save it by calling `self.do_query_meta()` and `self._save_metadata()`.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with metadata for each trace.
        """
        if self._meta is not None:
            return self._meta
        
        if self.config.use_cache and os.path.exists(self.meta_filename):
            df = pd.read_parquet(self.meta_filename)
            df['START'] = df['START'].apply(_format_to_datetime)
            df['END'] = df['END'].apply(_format_to_datetime)
            self._meta = df
        elif self.config.use_cache:
            logging.info(f'Cannot read {self.meta_filename}. I will kindly create it for you.')
            self._meta = self.do_query_meta()
            if not self._meta.empty:
                self._save_metadata()
        else:
            self._meta = self.do_query_meta()

        self.set_dict_for_trace_filenames()

        return self._meta 

    @meta.setter
    def meta(self, val):
        """
        Used to specify a different meta for this class.

        Parameters
        ----------
        val : pandas.DataFrame
            DataFrame containing metadata for each trace.
        """
        self._meta = val
        self._df_traces = None
        self._meta.reset_index(drop=True, inplace=True)
        self.set_dict_for_trace_filenames()

    def set_dict_for_trace_filenames(self):
        """
        Sets unique identifiers for trace filenames. This must be overriden.
        """
        return

    @property
    def index(self):
        """
        Alias for meta for backward compatibility.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with metadata for each trace.
        """
        return self.meta

    @index.setter
    def index(self, value):
        """
        Alias for setting meta for backward compatibility.

        Parameters
        ----------
        value : pandas.DataFrame
            DataFrame containing metadata for each trace.
        """
        self.meta = value

    @property
    def meta_filename(self):
        """
        Full path for the index filename.

        Returns
        -------
        str
            Full path for the index filename.
        """
        return f'{self.data_dir}/{self.config.meta_filename}_{self._unique_for_meta_filename}.parquet'

    @property
    def _unique_for_meta_filename(self):
        """
        Generates a unique identifier for the meta filename based on start and stop timestamps.

        Raises
        ------
        NotImplementedError
            This method should be overloaded in the inherited class.
        """
        raise NotImplementedError('Please overload do_query_meta in an inherited class.')

    @property
    def data_dir(self):
        """
        Path for the raw data directory, including dataset_name().

        Returns
        -------
        str
            Path for the raw data directory.
        """
        return f'{self.config.dataset_dir}/{self.config.dataset_name}'

    @property
    def processed_dir(self):
        """
        Path for the processed data directory, including dataset_name().

        Returns
        -------
        str
            Path for the processed data directory.
        """
        raise DeprecationWarning('Do not use processed dir yet!')
        return f'{self.config.processed_dir}/{self.config.dataset_name}'

    def do_query_meta(self):
        """
        Code to generate the metadata of the dataset. Must be overloaded!

        Returns
        -------
        pandas.DataFrame
            A DataFrame with metadata for each trace.
        
        Raises
        ------
        NotImplementedError
            This method should be overloaded in the inherited class.
        """
        raise NotImplementedError('Please overload do_query_meta in an inherited class.')

    def do_query_trace(self, row):
        """
        Code to generate the log trace between two timestamps. Must be overloaded!

        Parameters
        ----------
        row : dict
            Dictionary with the selected row in meta data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with trace data.

        Raises
        ------
        NotImplementedError
            This method should be overloaded in the inherited class.
        """
        raise NotImplementedError(f'Please overload do_query_trace in {self.__class__.__name__} class.')

    def _save_metadata(self):
        """
        Save self.meta in self.meta_filename cache file.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        self.meta.to_parquet(self.meta_filename, index=False, engine="pyarrow")

    def load_trace(self, idx, meta=None, cols=None, event_cols=None, save=True, use_cache=True):
        """
        Load a raw trace based on its ID in the metadata.

        If the cache file doesn't exist, it tries to generate and save it by calling self.do_query_trace() and self.save_trace()

        Parameters
        ----------
        idx : int
            The row number in the metadata dataframe.
        meta : pandas.DataFrame, optional
            If not provided, self.meta is used.
        cols : list, optional
            Restrict the dataframe to cols. This value can be set in the constructor too.
        event_cols : list, optional
            Create a new 'event' column by concatenating event_cols. This value can be set in the constructor too.
        save : bool, default True
            Save the cache file if it doesn't exist AND if meta is not set.
        use_cache : bool, default True
            If False, don't read the cache file and force a call to do_query_trace.

        Returns
        -------
        pandas.DataFrame
            DataFrame with trace data.
        """
        meta_to_use = meta if meta is not None else self.meta

        if not event_cols and self.config.event_cols:
            event_cols = self.config.event_cols

        if not cols and self.config.cols:
            cols = self.config.cols

        if self.config.use_cache and use_cache:
            filename = self.trace_filename(idx, meta_to_use)
            if os.path.exists(filename):
                if isinstance(event_cols, list) and cols == ['event']:
                    df = pd.read_parquet(filename, columns=event_cols, engine="pyarrow")
                else:
                    df = pd.read_parquet(filename)
                df.fillna('', inplace=True)
            else:
                logging.info(f'{filename} does not exist. I will query for you.')
                row = meta_to_use.loc[idx]
                df = self.do_query_trace(row)
                
                if save and meta is None:
                    os.makedirs(self.data_dir, exist_ok=True)
                    df.to_parquet(filename, index=False, engine="pyarrow")
                    logging.info(f'Cache file {self.config.trace_prefix} created with {len(df)} rows')
                else:
                    logging.info(f'I will NOT save {filename} because meta is user provided')
        else:
            # No cache at all
            row = meta_to_use.loc[idx]
            df = self.do_query_trace(row)

        if event_cols:
            df['event'] = df[event_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)

        if cols:
            # print(cols)
            df = df[cols]

        return df

    def trace_filename(self, idx, meta=None):
        """
        Returns the cached trace filename for a given index number.

        Parameters
        ----------
        idx : int
            The row number in the metadata dataframe.
        meta : pandas.DataFrame, optional
            If not provided, self.meta is used.

        Returns
        -------
        str
            Filename for the cached trace file.
        """
        meta = meta if meta is not None else self.meta
        timestamp_str = self._dict_for_trace_filenames[idx]
        suffix = '' if not self.config.trace_suffix else f'{self.config.trace_suffix}'

        if not self._trace_hash:
            md5bytes = hashlib.md5(''.join(self.filtered_trace_query()).encode('utf-8')).digest()
            self._trace_hash = base64.urlsafe_b64encode(md5bytes).decode('ascii')[:5]

        return f'{self.data_dir}/{self._trace_hash}_{timestamp_str}{suffix}.parquet'

    def traces(self, meta=None, cols=None, use_cache=None, warn=False):
        """
        Load and concatenate traces from the metadata.

        Parameters
        ----------
        meta : pandas.DataFrame, optional
            If not provided, self.meta is used.
        cols : list, optional
            Restrict the dataframe to cols.
        use_cache : bool, optional
            Default is self.config.use_cache
        warn: bool|func, optional
            If True, throw log warnings to show the progress

        Returns
        -------
        pandas.DataFrame
            Concatenated DataFrame with trace data.
        """
        if isinstance(warn, bool) and warn:
            log = logging.warning
        else:
            log = lambda x: x

        if not use_cache:
            use_cache = self.config.use_cache

        log(f"Loading {len(self.meta)} traces")
        if self._df_traces is None:
            df_meta = meta if meta is not None else self.meta
            # frames = [
            #     self.load_trace(idx, cols=cols, use_cache=use_cache)
            #     for idx in meta.index.values
            # ]
            frames = []
            len_meta = len(df_meta)
            for count, idx in zip(range(len_meta), df_meta.index.values):
                t = self.load_trace(idx, cols=cols, use_cache=use_cache)
                t["trace_id"] = idx
                frames.append( t )
                if count % (len_meta // 20) == 0:
                    log(f"{count} of {len_meta} loaded")
            self._df_traces = pd.concat(frames, ignore_index=True)
            log("All traces loaded!")
        return self._df_traces

    def preload_traces(self):
        """
        Preloads traces for faster access.

        Loads traces in chunks to provide a progress indicator.

        Returns
        -------
        None
        """
        raise NotImplementedError("This method needs refactor")
        chunk = int(len(self.meta) / 10.0 + 0.5)
        for i in self.meta.index.values:
            if not os.path.exists(self.trace_filename(i)):
                trace = self.load_trace(i)
                print(f'... Preloading trace {i} with {len(trace)} events')
            if i % chunk == chunk - 1:
                perc = int(100.0 * i / len(self.meta) + 0.5)
                print(f'{perc}% of {len(self.meta)} traces preloaded')

    def signature(self, **kwargs):
        """
        When inherited, add the new signatures to the right:

        return f"{super().signature(**kwargs)}-{myCurrentSignature}"
        """
        return self.__class__.__name__

class ElasticsearchQueryDataset(ElianaDataset):
    """
    Parlogan dataset specialized in Elasticsearch queries.

    
    """
    def __init__(self, **kwargs):
        """
        Main parameter: **config
        
        Parameters
        ----------
        config : dict
            A configuration dictionary containing key-value pairs for the dataset setup.

        Attributes
        ----------
        start_timestamp : str
            Start timestamp for the dataset instance.
        stop_timestamp : str
            Stop timestamp for the dataset instance.
        trace_query : str
            Query string for filtering traces.
        meta_query : str
            Query string for filtering the metadata.

        Required config values
        ----------------------
        start_timestamp : str
            The start timestamp for the dataset.
        stop_timestamp : str
            The stop timestamp for the dataset.

        Optional config values
        ----------------------
        trace_filters : list
            Filters to be appended to `self.trace_query`.

        Default config values
        ---------------------
        'meta_query' : str
            Default is ''.
        'trace_query' : str
            Default is ''.
        'trace_filters' : list
            Default is an empty list.

        Methods
        -------
        start_timestamp
            Getter and setter for the start timestamp.
        stop_timestamp
            Getter and setter for the stop timestamp.
        add_trace_filter(filter)
            Adds filters to the trace query.
        filtered_trace_query(**kwargs)
            Returns the trace query with applied filters.
        
        """
        super().__init__(**kwargs)
        required_keys = ['start_timestamp', 'stop_timestamp']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f'{key} must be defined in config')

        self._start_timestamp = kwargs['start_timestamp']
        self._stop_timestamp = kwargs['stop_timestamp']
        self._trace_filters = kwargs.get('trace_filters', [])
        self.trace_query = kwargs.get('trace_query', '')
        self.meta_query = kwargs.get('meta_query', '')        

    @property
    def start_timestamp(self):
        """
        Start timestamp for this dataset instance.

        Returns
        -------
        str
            Start timestamp.
        """
        return self._start_timestamp

    @start_timestamp.setter
    def start_timestamp(self, val):
        """
        Sets the start timestamp for this dataset instance.

        Parameters
        ----------
        val : str
            Start timestamp.
        """
        self._start_timestamp = val

    @property
    def stop_timestamp(self):
        """
        Stop timestamp for this dataset instance.

        Returns
        -------
        str
            Stop timestamp.
        """
        return self._stop_timestamp

    @stop_timestamp.setter
    def stop_timestamp(self, val):
        """
        Sets the stop timestamp for this dataset instance.

        Parameters
        ----------
        val : str
            Stop timestamp.
        """
        self._stop_timestamp = val

    def add_trace_filter(self, filter):
        """
        Add filters to trace query: (trace_query) -(filter1 OR filter2 ...)

        Parameters
        ----------
        filter : str
            Filter to be added to the trace query.
        """
        self._trace_filters.append(filter)
        self._trace_hash = False

    def filtered_trace_query(self, **kwargs):
        """
        Returns (self.trace_query) -(filter1 OR filter2 ...)

        If one or more parameters are passed, they are used to format trace_query.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs for formatting the trace query.

        Returns
        -------
        str
            Trace query with applied filters.
        """
        query = f'({self.trace_query}) '
        if self._trace_filters:
            query += '-(' + ' '.join([f'({x})' for x in self._trace_filters]) + ')'
            logging.debug(query)
        return query.format(**kwargs) if kwargs else query

    def set_dict_for_trace_filenames(self):
        """
        Sets a dictionary with unique identifiers for trace filenames based on the START time in 
        each row of the metadata.

        This method generates a unique identifier for each trace file using the START timestamp 
        from the metadata (`self.meta`). The identifier is a formatted string of the timestamp 
        up to microseconds, which ensures that each trace file has a unique and time-based name.

        The resulting identifiers are stored in the `_dict_for_trace_filenames` attribute, 
        which is a dictionary mapping the index of each row in the metadata to its corresponding 
        unique filename identifier.

        Returns
        -------
        dict
            A dictionary mapping each index in the metadata to its unique trace filename identifier.

        Example
        -------
        If the metadata (`self.meta`) contains the following START times:
        ```
        idx   START
        0     2023-07-17 10:15:30.123
        1     2023-07-17 11:20:45.654
        ```
        The `_dict_for_trace_filenames` attribute will be set to:
        ```
        {
            0: '2023-07-17T10-15-30-123',
            1: '2023-07-17T11-20-45-654'
        }
        ```

        This is used later by ElianaDataset.trace_filename to generate names likes:
        5RPei_2019-03-02T00-43-48-280.parquet
        """
        if "START" in self.meta.columns:
            self._dict_for_trace_filenames = {
                idx: st.strftime('%Y-%m-%dT%H-%M-%S-%f')[:23]
                for idx, st in self.meta['START'].items()
            }
        else:
            self._dict_for_trace_filenames = {}
        return self._dict_for_trace_filenames

    @property
    def _unique_for_meta_filename(self):
        """
        Generates a unique identifier for the meta filename based on start and stop timestamps.

        Returns
        -------
        str
            Unique identifier for the meta filename.
        """
        return f"{self.config.start_timestamp}_{self.config.stop_timestamp}".replace(':', '-')
    


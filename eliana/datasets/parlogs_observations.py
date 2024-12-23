import logging
from urllib.request import urlretrieve
import os
import pandas as pd
from eliana.datasets import ElianaDataset

def _repopulate_config_and_allowed(allowed_values):
    def decorator(func):
        def wrapper(self, val):
            values_to_check = [val] if isinstance(val, str) else val
            if values_to_check and any(x not in allowed_values for x in values_to_check):
                raise ValueError(f"{func.__name__} must be one or more of {allowed_values}")
            res = func(self, val)
            setattr(self.config, func.__name__, val)
            return res
        return wrapper
    return decorator

class ParlogsObservations(ElianaDataset):
    """
    Public Dataset with VLTI logs from 2019.

    This dataset includes logs from instruments, telescopes, and subsystems, and allows filtering based on system, period, and source criteria.

    See in https://huggingface.co/datasets/Paranal/parlogs-observations

    Attributes
    ----------
    available_systems : list
        List of available systems.
    available_periods : list
        List of available periods.
    available_sources : list
        List of available sources.
    system : str
        The system for which logs are being queried.
    period : str
        The period for which logs are being queried.
    source : list of str
        The source of the dataset being queried, default Instrument

    Parameters
    ----------
    **kwargs : dict
        Additional configuration parameters, including but not limited to:
        - system: str, optional, the name of the system to query logs for.
        - period: str, optional, the period to query logs for.
        - source: list of str, optional, the sources to include in the dataset.
        - dataset_dir: str, optional, location of cache files. default=data/raw
    """

    available_systems = ["PIONIER", "GRAVITY", "MATISSE"]
    available_periods = ["1d", "1w", "1m", "6m"]
    available_sources = ["Instrument", "Subsystems", "Telescopes", "All"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.use_cache = False
        self._df_traces = None

        # Set default values if not provided
        defaults = {
            'system': self.available_systems[0],
            'period': self.available_periods[0],
            'source': self.available_sources[0],
            'event_cols': ['logtype', 'procname', 'keywname', 'keywvalue', 'logtext']
        }

        # Update defaults with provided kwargs and set attributes
        for key, default in defaults.items():
            setattr(self, key, kwargs.get(key, default))
            if getattr(self.config, key, None):
                setattr(self, key, getattr(self.config, key))
            else:
                setattr(self.config, key, default)
        # # Update defaults with provided kwargs and set attributes
        # for key, default in defaults.items():
        #     value = kwargs.get(key, default)
        #     setattr(self, key, value)
        #     setattr(self.config, key, getattr(self.config, key, value))

    @property
    def system(self):
        return self._system

    @system.setter
    @_repopulate_config_and_allowed(available_systems)
    def system(self, val):
        self._system = val

    @property
    def period(self):
        return self._period

    @period.setter
    @_repopulate_config_and_allowed(available_periods)
    def period(self, val):
        self._period = val

    @property
    def source(self):
        return self._source

    @source.setter
    @_repopulate_config_and_allowed(available_sources)
    def source(self, val):
        if isinstance(val, str):
            val = [val]
        self._source = val
        self._df_traces = None

    def do_query_meta(self):
        """
        Generates the metadata for the dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the generated metadata.
        """
        meta_file = f"{self.system}-{self.period}-meta.parket"
        self.preload_public_file(meta_file)
        dataset = pd.read_parquet(f"{self.data_dir}/{meta_file}")
        dataset.rename(columns={'Aborted': 'USER_ABORT'}, inplace=True)
        dataset['TPL_EXEC'] = 'STOP'
        dataset.loc[ (dataset['ERROR'] ) | (dataset['USER_ABORT'] ), 'TPL_EXEC'] = 'ABORT'
        return dataset

    def do_query_trace(self, row):
        """
        Generates the trace for the dataset based on the row of metadata.

        Parameters
        ----------
        row : pandas.Series
            Row of metadata.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the generated trace.
        """
        df_traces = self.traces()
        df = df_traces[df_traces["trace_id"] == row.name]
        df.reset_index(drop=True, inplace=True)
        return df

    def traces(self, meta=None, cols=None, use_cache=None, warn=False):
        """
        Loads and concatenates traces from the metadata.

        Parameters
        ----------
        meta : pandas.DataFrame, optional
            If not provided self.meta is used.
        cols : list, optional
            Restrict the dataframe to cols.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the traces.
        """
        self.preload_traces()
        df = self._df_traces.copy()

        event_cols = self.config.event_cols
        if not cols and self.config.cols:
            cols = self.config.cols
        if event_cols:
            df['event'] = df[event_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)

        if cols:
            df = df[cols]
        return df

    def preload_traces(self):
        """
        Preloads traces from parket files.
        """
        if self._df_traces is None:
            all_traces = []
            for file in self._source_files():
                self.preload_public_file(file)
                all_traces.append(pd.read_parquet(f"{self.data_dir}/{file}"))
            self._df_traces = pd.concat(all_traces)
            self._df_traces.sort_values('@timestamp', inplace=True)
            self._df_traces.reset_index(drop=True, inplace=True)

    def _source_files(self):
        """
        Returns the list of source files based on the source attribute.
        """
        base_name = f"{self.system}-{self.period}"

        file_map = {
            "Instrument": [f"{base_name}-traces.parket"],
            "Telescopes": [f"{base_name}-traces-TELESCOPES.parket"],
            "Subsystems": [f"{base_name}-traces-SUBSYSTEMS.parket"],
            "All": [
                f"{base_name}-traces.parket",
                f"{base_name}-traces-TELESCOPES.parket",
                f"{base_name}-traces-SUBSYSTEMS.parket"
            ]
        }

        source_files = []
        for source in self.source:
            # source_files.extend(file_map[split])
            source_files += file_map[source]

        return list(set(source_files))

    def preload_public_file(self, file, PATH=None):
        if not PATH:
            PATH = self.data_dir
        REPO_URL='https://huggingface.co/datasets/Paranal/parlogs-observations/resolve/main/data'
        if not os.path.exists(f'{PATH}/{file}'):
            logging.warning(f"Downloading {PATH}/{file}")
            logging.warning(f"from {REPO_URL}/{file}")
            os.makedirs(PATH, exist_ok=True)
            urlretrieve(f'{REPO_URL}/{file}', f'{PATH}/{file}')

    def signature(self, **kwargs):
        return f"{super().signature(**kwargs)}-{self.system}-{'-'.join(self.source)}-{self.period}"


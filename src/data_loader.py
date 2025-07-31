import pandas as pd
import operator
import re

# Mapping string operators to Python functions
OPS = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}

DELIMITERS = {
    'comma': ',',
    'tab': '\t',
    'whitespace': r'\s+'
}

def apply_filters(df, filters):
    for f in filters:
        col = f['column']
        op = OPS[f['operator']]
        val = f['value']
        df = df.loc[op(df[col], val)]
    return df

def extract_column(df, pattern, method='mean'):
    cols = df.filter(regex=pattern).columns
    if len(cols) == 0:
        raise ValueError(f"No columns match pattern: {pattern}")
    elif len(cols) == 1:
        return df[cols[0]]
    else:
        if method == 'mean':
            return df[cols].mean(axis=1)
        elif method == 'sum':
            return df[cols].sum(axis=1)
        elif method == 'first':
            return df[cols[0]]
        else:
            raise ValueError(f"Unsupported collapse method: {method}")

def load_dataset(config):
    """Function that reads specifications from the config file to load in filesets."""
    
    delimiter = DELIMITERS.get(config.get('delimiter', 'whitespace'), r'\s+')
    skiprows = config.get('skiprows', 0)
    path = config['path']

    df = pd.read_csv(path, delimiter=delimiter, skiprows=skiprows)

    if 'filters' in config:
        df = apply_filters(df, config['filters'])

    out = pd.DataFrame()

    # Optional UTC/time column
    if 'columns' in config and 'UTC' in config['columns']:
        out['UTC'] = df['UTC']
    elif 'time_column' in config:
        out['UTC'] = df[config['time_column']]

    drop_rows = config.get('drop_rows', [])
    if isinstance(drop_rows, int):
        drop_rows = [drop_rows]

    # Drop all rows from a certain index onward, if specified
    drop_rows_from = config.get('drop_rows_from')
    if drop_rows_from is not None:
        drop_rows.extend(range(drop_rows_from, len(df)))

    # Sanity filter for valid indices
    valid_drop_rows = [i for i in drop_rows if i < len(df)]
    if valid_drop_rows:
        df = df.drop(index=df.index[valid_drop_rows]).reset_index(drop=True)

    # Handle regex-based extractions
    regex_config = config.get('regex_columns', {})

    # Process magnitude columns
    if 'magnitude' in regex_config:
        mag_cols = df.filter(regex=regex_config['magnitude']).copy()

        if 'mag_column' in config:
            first_col = mag_cols.columns[0]
            out[config['mag_column']] = mag_cols[first_col]
        else:
            # Keep all matched columns if no rename is specified
            for col in mag_cols.columns:
                out[col] = mag_cols[col]

    # Process uncertainty/error columns
    if 'error' in regex_config:
        err_cols = df.filter(regex=regex_config['error']).copy()
        if 'err_column' in config:
            first_col = err_cols.columns[0]
            out[config['err_column']] = err_cols[first_col]
        else:
            # Use default column name
            out['mag_err'] = extract_column(df, regex_config['error'])

    # Add optional source tag
    out['source'] = config['name']

    return out

def load_all_datasets(dataset_configs):
    dfs = []
    for config in dataset_configs:
        try:
            df = load_dataset(config)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Failed to load dataset '{config['name']}': {e}")
    return pd.concat(dfs, ignore_index=True)
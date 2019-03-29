import datetime

def get_continuous_blocks(orig_df, threshold=1):
    """ Cuts the dataframe into continuous "threshold" minute data.

    Parameters
    ----------
    orig_df     : pd.DataFrame()
        Data to cut.
    threshold   : int
        Interval of each block. (in minutes)

    Returns
    -------
    list
        List of start and end times of each block.

    """

    blocks = []
    df = orig_df.copy()
    
    df["time_diff"] = df.index
    
    # time_diff contains the difference in time between two rows
    df["time_diff"] = (df['time_diff'] - df['time_diff'].shift())
    
    # List of datetimeindices where time_diff != 1 min
    start_indices = df.loc[df["time_diff"] != datetime.timedelta(minutes=1)].index
    
    for i in range(1, len(start_indices)):
        st = start_indices[i-1]
        et = start_indices[i] - datetime.timedelta(minutes=threshold)
        if st >= et:
            continue
        blocks.append((st, et))
        
    final_st = start_indices[-1]
    final_et = df.index[-1]
    
    if final_st < final_et:
        blocks.append((final_st, final_et))
    
    return blocks


def get_single_state_diff(orig_df, threshold=1):
    """ Cuts a dataframe with continouous data and returns
    start and end times of chunks with single state changes only.

    Parameters
    ----------
    orig_df     : pd.DataFrame()
        Data to cut.
    threshold   : int
        Interval of each single state change block. (in minutes)

    Returns
    -------
    list
        List of start and end times of each block.

    """
    
    blocks = []
    df = orig_df.copy()
    df['state_diff'] = df.state.diff()
    df = df.dropna()
    
    str_diff = df['state_diff'].abs().astype(int).astype(str)    
    df['num_state_changes'] = str_diff.str.len() - str_diff.str.count('0')
    
    # Gets start indices of all rows where the number of state changes is 1
    start_indices = df.loc[(df['num_state_changes'] == 1)].index

    if not start_indices.empty:    
        
        for i in range(1, len(start_indices)):
            st = start_indices[i-1]
            et = start_indices[i] - datetime.timedelta(minutes=threshold)
            blocks.append((st, et))

        final_st = start_indices[-1]
        final_et = df.index[-1]
        blocks.append((final_st, final_et))
    
    return blocks


def get_first_last_block_power(df, block_duration=5):
    """ Returns the avg power of first and last 5min of dataframe.

    Note
    ----
    1. df.loc[st:et].power.mean() --> ".power" is hardcoded.
    2. the first avg is not used.
    
    Parameters
    ----------
    df                  : pd.DataFrame()
        Power data.
    block_duration      : int
        Duration of interval of first and last block (in minutes)

    Returns
    -------
    float, float
        First and last "block_duration" min of avg df data.

    """
    
    st = df.index[0]
    et = st + datetime.timedelta(minutes=block_duration)
    first = df.loc[st:et].power.mean()
    
    et = df.index[-1]
    st = et - datetime.timedelta(minutes=block_duration)
    last = df.loc[st:et].power.mean()
    
    return first, last


def get_action_from_state_diff(current_state, prev_state):
    """ Returns the column which had its state changed.

    Parameters
    ----------
    current_state   : float
        The current state of HVAC zone.
    prev_state      : float
        The previous state of HVAC zone.

    Returns
    -------
    str
        Columns name which had its state changed.
    
    """
    
    i = 0
    state_diff = abs(current_state - prev_state)

    while state_diff > 0:
        
        if state_diff % 10 != 0:
            return 's%d_%d' % (i, current_state % 10)
        
        state_diff = state_diff / 10
        current_state = current_state / 10
        i += 1
    
    return False


def get_continuous_states(orig_df, threshold=1):
    df = orig_df.copy()
    df['state_diff'] = df.state.diff()
    start_idx = df.loc[(df['state_diff'] != 0)].index
    blocks = []
    for i in range(1, len(start_idx)):
        st = start_idx[i-1]
        et = start_idx[i] - datetime.timedelta(minutes=threshold)
        
        blocks.append((st, et))
    final_st = start_idx[-1]
    final_et = df.index[-1]
    blocks.append((final_st, final_et))
    return blocks

def get_five_min_blocks(orig_df, block_duration=5):
    df = orig_df.copy()
    
    blocks = []
    st = df.index[0]
    et = st+datetime.timedelta(minutes=block_duration)
    
    while st <= df.index[-1]:
        blocks.append((st, et))
        st = et+datetime.time_delta(minutes=1)
        et = st+datetime.timedelta(minutes=block_duration)
        
        if et > df.index[-1]:
            et = df.index[-1]
    return blocks

def get_five_min_blocks(orig_df, block_duration=5):
    df = orig_df.copy()
    final_et = df.index[-1]
    blocks = []
    st = df.index[0]
    et = st+datetime.timedelta(minutes=block_duration)

    if et>final_et:
        et = final_et
    
    
    while st <= final_et:
        blocks.append((st, et))
        st = et+datetime.timedelta(minutes=1)
        et = st+datetime.timedelta(minutes=block_duration)
        
        if et > final_et:
            et = final_et
    return blocks

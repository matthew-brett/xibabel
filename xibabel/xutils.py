""" Code utilities for xibabel package
"""

def merge(d1, d2):
    """ Recursive merge of dictionaries `d1` and `d2`

    Parameters
    ----------
    d1 : dict
    d2 : dict

    Returns
    -------
    dout : dict
        Recursive merge of `d1` and `d2`.  For each dictionry contained in
        `d1`, update keys from equivalent (if present) in `d2`, but
        recursively, so that sub-dictionaries are also merged (updated).

    Notes
    -----
    Initially from `https://stackoverflow.com/a/50441142/1939576`_
    """
    # Recursive dictionary merge (update).
    if isinstance(d1, dict) and isinstance(d2, dict):
        return {
            **d1, **d2,
            **{k: d1[k] if d1[k] == d2[k] else merge(d1[k], d2[k])
            for k in {*d1} & {*d2}}
        }
    return d2

__all__ = ['hashstr']

import hashlib

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%nr_bins

import hashlib, csv

def open_skip_first_line(path):
    f = open(path)
    next(f)
    return f

def open_csv_skip_first_line(path):
    f = csv.reader(open(path))
    next(f)
    return f

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%nr_bins

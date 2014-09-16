import hashlib, csv, math, os, pickle

HEADER="Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"

def open_with_first_line_skipped(path):
    f = open(path)
    next(f)
    return f

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_feats(row):
    feats = []
    for j in range(1, 14):
        field = 'I{0}'.format(j)
        value = row[field]
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value))**2)
            else:
                value = 'SP'+str(value)
        key = field + '-' + str(value)
        feats.append(key)
    for j in range(1, 27):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats

def gen_hashed_svm_feats(feats, nr_bins, coef=None):
    feats = [hashstr(feat, nr_bins) for feat in feats]
    feats = list(set(feats))
    feats.sort()
    if coef is not None:
        val = coef
    else:
        val = round(1/math.sqrt(float(len(feats))), 5)
    feats = ['{0}:{1}'.format(idx, val) for idx in feats]
    return feats

def read_freqent_feats(threshold=10):
    if os.path.exists('fc.trva.r1.p1.t10.pickle'):
        return pickle.load(open('fc.trva.r1.p1.t10.pickle', 'rb'))

    frequent_feats = {}
    for row in csv.DictReader(open('fc.trva.r1.p1.t10.txt')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats[row['Field']+'-'+row['Value']] = float(row['Ratio'])
    return frequent_feats

def gen_hashed_fm_feats(feats, nr_bins, coef=None):
    feats = [(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    feats.sort()
    if coef is not None:
        val = coef
    else:
        val = round(1/math.sqrt(float(len(feats))), 5)
    feats = ['{0}:{1}:{2}'.format(field, idx, val) for (field, idx) in feats]
    return feats

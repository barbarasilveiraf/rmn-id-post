import theano, cPickle, h5py, lasagne, random, csv, gzip                                                  
import numpy as np
import theano.tensor as T         


# convert csv into format readable by rmn code
def load_data(span_path, metadata_path):
    #x = csv.DictReader(gzip.open(span_path, 'rb'))
    x = csv.DictReader(open(span_path, 'r'))

    #print x

    wmap, cmap, bmap = cPickle.load(open(metadata_path, 'rb'))
    max_len = -1

    revwmap = dict((v,k) for (k,v) in wmap.iteritems())
    revbmap = dict((v,k) for (k,v) in bmap.iteritems())
    revcmap = dict((v,k) for (k,v) in cmap.iteritems())

    #print revbmap

    span_dict = {}
    span_ids = {}
    for row in x:
        #if row['subs'] != 'depression': 
            #print row
        #    continue
        text = row['spans'].split()
        #text = text.replace('nan', '').replace('null', '').replace('NA','').replace('Nan', '')
        if len(text) > max_len:
            max_len = len(text)
        key = '$$-'.join([row['subs'], row['user']])
        if key not in span_dict:
            span_dict[key] = []
            span_ids[key] = []
        try:
            span_dict[key].append([wmap[w] for w in text if w != 'nan' and w != 'null' and w != 'NA'])
            span_ids[key].append(row['id'])
        except Exception as e:
            print e
           # exit()

    print "montei spans"
    
    span_data = []
    for key in span_dict:
        #print key
        book, c1 = key.split('$$-')
        book = np.array([revbmap[book], ]).astype('int32')
        chars = np.array([revcmap[c1], ]).astype('int32')

        # convert spans to numpy matrices 
        spans = span_dict[key]
        s = np.zeros((len(spans), max_len)).astype('int32')
        m = np.zeros((len(spans), max_len)).astype('float32')
        for i in range(len(spans)):
            curr_span = spans[i]
            s[i][:len(curr_span)] = curr_span
            m[i][:len(curr_span)] = 1.

        span_data.append([book, chars, s, m, span_ids[key]])
    print "montei span data"
    return span_data, max_len, wmap, cmap, bmap


def generate_negative_samples(num_traj, span_size, negs, span_data):
    inds = np.random.randint(0, num_traj, negs)
    neg_words = np.zeros((negs, span_size)).astype('int32')
    neg_masks = np.zeros((negs, span_size)).astype('float32')
    for index, i in enumerate(inds):
        rand_ind = np.random.randint(0, len(span_data[i][2]))
        neg_words[index] = span_data[i][2][rand_ind]
        neg_masks[index] = span_data[i][3][rand_ind]

    return neg_words, neg_masks


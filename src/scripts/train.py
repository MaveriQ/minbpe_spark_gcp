import regex as re
from ..utils.helper_functions import get_counts, merge
from pyspark.rdd import RDD
from time import time

def train(rdd: RDD, 
          config: dict,
          compiled_pattern: str,
          morphtable: dict,
          verbose: bool=False) -> tuple:
    
    assert config['vocab_size'] >= 256
    num_merges = config['vocab_size'] - 256
    start_time = time()
    
    if config.lowercase:
        # text = text.lower()
        rdd = rdd.map(lambda x: {config['text_column']: x[config['text_column']].lower()})
        
    # split the text up into text chunks
    # text_chunks = re.findall(self.compiled_pattern, text)
    text_chunks = rdd.map(lambda x: re.findall(compiled_pattern,x[config['text_column']]))
    
    # chunks without morphemes
    if config.morphemize:
        text_chunks = text_chunks.map(lambda e: e.strip()).filter(lambda e: e not in morphtable)

    # input text preprocessing
    # ids = [list(ch.encode("utf-8")) for ch in text_chunks]
    ids = text_chunks.map(lambda x: [list(ch.encode("utf-8")) for ch in x])

    # iteratively merge the most common pairs to create new tokens
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
    for i in range(num_merges):
        # count the number of times every consecutive pair appears
        # stats = {}
        # for chunk_ids in ids:
            # passing in stats will update it in place, adding up counts
            # get_stats(chunk_ids, stats)
        stats = ids.map(lambda x: [get_counts(y) for y in x]).flatMap(lambda x: [y for y in x if len(y)!=0])
        # find the pair with the highest count
        # pair = max(stats, key=stats.get)
        pair = stats.flatMap(lambda line: line).map(lambda pair: (pair,1)).reduceByKey(lambda x,y: x+y).map(lambda x: (x[1],x[0])).sortByKey(False).take(1)
        pair = pair[0][1]
        # mint a new token: assign it the next available id
        idx = 256 + i
        # replace all occurrences of pair in ids with idx
        # ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
        ids = ids.map(lambda x: merge(x,pair,idx)).collect()
        
        # save the merge
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # time for iteration
        end_time = time()
        iter_time = end_time-start_time
        start_time = time()
        
        # prints
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences and the iteration took {round(iter_time,3)} seconds.")

    # save class variables
    # self.merges = merges # used in encode()
    # self.vocab = vocab   # used in decode()
    return merges,vocab
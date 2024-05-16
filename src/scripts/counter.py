from morphpiece import MorphPiece
from pyspark.rdd import RDD

def get_counts(rdd: RDD,
               config: dict,
               m: MorphPiece,
               ) -> dict:
    
    text_column = config['text_column']
    count_type = config['count_type']

    rdd.map(lambda e: m.tokenize(e[text_column])).flatMap(
        lambda x: tuple(x.items())).reduceByKey(lambda x, y: (x+y)).collect()

    if count_type == "morph":
        return m.counter_morph
    elif count_type == "bpe_split":
        return m.counter_bpe_split
    elif count_type == "bpe_nonsplit":
        return m.counter_bpe_nonsplit
    else:
        raise ValueError()

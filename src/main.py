# Import Python libraries.
from typing import Optional
import argparse
import regex as re

# Import third party libraries.
from codetiming import Timer
from morphpiece import MorphPiece

# Import local modules.
from src.utils import start_spark, timer_args
from src.scripts import read_file, train, get_counts, save_to_gcs

GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def run(app_name: Optional[str],
        known_args: argparse.Namespace,
        ) -> None:
    """
    :param app_name: Spark App Name,
    :param known_args: Known Arguments
    :return: None
    """
    bucket = known_args.temp_bq_bucket,
    file_uri = known_args.parquet_uri,
    pattern = known_args.pattern,
    count_type = known_args.count_type
    output_uri = known_args.output_uri
    
    config = {
        'vocab_size': known_args.vocab_size,
        'text_column': known_args.text_column,
        'lowercase': known_args.lowercase,
        'morphemize': known_args.morphemize,
        'count_type': known_args.count_type,
    }

    total_time = Timer(**timer_args("Total run time"))
    total_time.start()
    
    m = MorphPiece()

    compiled_pattern = re.compile(pattern)

    with Timer(**timer_args('Spark Connection')):
        spark = start_spark(app_name=app_name,
                            bucket=bucket)

    with Timer(**timer_args('Read File From GCS')):
        text_rdd = read_file(spark=spark,
                             file_uri=file_uri,)

    if known_args.task == 'train':
        with Timer(**timer_args('Train BPE Tokenizer')):
            merge, vocab = train(rdd=text_rdd,
                                 config=config,
                                 compiled_pattern=compiled_pattern,
                                 morphtable=m.morphtable,
                                 verbose=True)

        with Timer(**timer_args('Save BPE Tokenizer')):
            status = save_to_gcs(payload={'merge': merge,
                                          'vocab': vocab},
                                 bucket=output_uri)
            
    elif known_args.task == 'counter':

        with Timer(**timer_args('Counter')):
            counts = get_counts(rdd=text_rdd,
                                m=m,
                                config=config
                                )

        with Timer(**timer_args('Save Counter Information')):
            status = save_to_gcs(payload={'count': counts},
                                 bucket=output_uri)
    else:
        raise ValueError()

    total_time.stop()
    print(Timer.timers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--parquet-uri',
        type=str,
        dest='parquet_uri',
        required=True,
        help='URI of the GCS bucket with parquet files, for example, gs://bucket_name/file_name')

    parser.add_argument(
        '--output-uri',
        type=str,
        dest='output_uri',
        required=True,
        help='URI of the GCS bucket for output files, for example, gs://bucket_name/file_name')
    
    parser.add_argument(
        '--project',
        type=str,
        dest='project_id',
        required=True,
        help='GCP Project ID')

    parser.add_argument(
        '--temp-bq-bucket',
        type=str,
        dest='temp_bq_bucket',
        required=True,
        help='Name of the Temp GCP Bucket -- DO NOT add the gs:// Prefix')

    parser.add_argument(
        "--pattern",
        type=str,
        dest='pattern',
        required=False,
        default=GPT4_PATTERN,
        help="Regex Pattern to split the text on. Default is GPT4_PATTERN."
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        dest='vocab_size',
        required=False,
        default=32000,
        help="Size of the Vocabulary to train the BPE model on. Default is 32000."
    )

    parser.add_argument(
        "--text_column",
        type=str,
        dest='text_column',
        default='text',
        help="Name of the Text Column in the dataset. Default is 'text'."
    )

    parser.add_argument(
        "--task",
        type=str,
        dest='task',
        default='counter',
        choices=['train', 'counter'],
        help="Name of the Text Column in the dataset. Default is 'text'."
    )

    parser.add_argument(
        "--count_type",
        type=str,
        dest='count_type',
        default='text',
        help="Name of the Text Column in the dataset. Default is 'text'."
    )
    
    parser.add_argument(
        "--lowercase",
        type=str,
        dest='lowercase',
        default=False,
        help="Make the text lowercase. Default is False."
    )
    
    parser.add_argument(
        "--morphemize",
        type=str,
        dest='morphemize',
        default=False,
        help="Filter out morphemes from the text. Default is False."
    )

    known_args, pipeline_args = parser.parse_known_args()

    run(app_name=f"serverless-spark-{known_args.task}-pipeline",
        args=known_args,
        )

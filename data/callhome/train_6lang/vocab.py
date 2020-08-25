"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm


def make_vocab(args):
    """Constructs vocabulary.
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    with open(args.input, encoding='utf-8') as f:
        for l in f:
            words = l.strip().split()
            word2cnt.update(Counter(words))
    with open(args.output, 'w', encoding='utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{} {}\n".format(word, cnt))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, dest='output')
    parser.add_argument('--input', type=str, dest='input')
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    make_vocab(args)
    # pre_processing(args.src_path, args.src_vocab)
    logging.info("Done")

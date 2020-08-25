"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict


def main(args):
    word2cnt = Counter()
    num = 0
    with open(args.text, encoding='utf-8') as f:
        for i, l in enumerate(f):
            try:
                uttid, words = l.strip().split(maxsplit=1)
            except:
                num += 1
                continue
                # print(l)
            if 'ma' in uttid or 'ja' in uttid:
                continue
            word2cnt.update(Counter(words.split()))
    print('drop {}/{}'.format(num, i))
    with open(args.output, 'w', encoding='utf-8') as fout:
        for word, _ in word2cnt.most_common():
            fout.write(u"{} {}\n".format(word, ' '.join(word) + ' |'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, dest='output')
    parser.add_argument('--text', type=str, dest='text')
    args = parser.parse_args()
    # Read config

    main(args)
    logging.info("Done")

"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict


def load_lexicon(path):
    word2phones = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            word, phones = line.strip().split(maxsplit=1)
            if phones[-1] != '|':
                phones = phones + ' |'
            word2phones[word] = phones

    return word2phones


def load_text(path):
    uttid2text = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            try:
                uttid, text = line.strip().split(maxsplit=1)
            except:
                print(line)
                continue
            uttid2text[uttid] = text

    return uttid2text


def main(args):
    word2phones = load_lexicon(args.lexicon)
    uttid2text = load_text(args.text)
    num = 0
    unk = set()
    with open(args.tsv) as f, open(args.ltr, 'w') as fw, open(args.tsv + '_new', 'w') as fw2:
        for i, line in enumerate(f):
            try:
                path, tmp = line.strip().split()
                uttid = path.split('/')[-1].split('.')[0]
                text = uttid2text[uttid]
            except:
                num += 1
                continue
            list_trans = []
            for word in text.split():
                if word not in word2phones.keys():
                    if not set(word).difference(set('qwertyuiopasdfghjklzxcvbnmóáñéúÓ')):
                        phones = ' '.join(word) + ' |'
                    else:
                        match(word2phones, word)
                else:
                    phones = word2phones[word]

                list_trans.append(phones)
            res = ' '.join(list_trans)
            fw.write(res + '\n')
            fw2.write(line)
    print('drop {}'.format(num))
    print(unk, len(unk))

num = 0
def match(lexicon, word):
    global num
    if word in lexicon.keys():
        return lexicon[word]
    for i in range(-1, -len(word), -1):
        if word[:i] in lexicon.keys():
            return lexicon[word[:i]] + match(lexicon, word[i:])
    num+= 1
    print('not found', word, num)
    return 'SPN'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tsv', type=str, dest='tsv')
    parser.add_argument('--ltr', type=str, dest='ltr')
    parser.add_argument('--lexicon', type=str, dest='lexicon')
    parser.add_argument('--text', type=str, dest='text')
    args = parser.parse_args()
    # Read config
    # python tsv2ltr.py --tsv valid.tsv --ltr valid_.ltr --lexicon lexicon.txt --text all_dev_text

    main(args)
    logging.info("Done")

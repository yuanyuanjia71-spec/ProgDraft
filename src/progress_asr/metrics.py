"""Corpus WER/CER normalization used by the recorded Final Test results."""
import re
import unicodedata

def normalize(text):
    for token,punctuation in {'<COMMA>':',','<PERIOD>':'.','<QUESTIONMARK>':'?','<EXCLAMATIONPOINT>':'!'}.items():text=text.replace(token,punctuation)
    text=re.sub(r'<[^>]*>','',text)
    text=unicodedata.normalize('NFKC',text).lower()
    # English orthography; keep apostrophes inside words, remove other punctuation.
    text=text.replace('’',"'")
    text=''.join(c if c.isalnum() or c.isspace() or c=="'" else ' ' for c in text)
    return ' '.join(text.split())


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

import json

import regex as re

import spacy.parts_of_speech as spos
from bootleg.end2end.bootleg_annotator import BootlegAnnotator
# print(ann.label_mentions("בא לי לאכול במסעדת מצה איזו עוגית אוראו")["titles"])

from pathlib import Path
from bootleg.utils.utils import load_yaml_file
import time
import string
from bootleg.utils.utils import get_lnrm
import ujson
from tqdm import tqdm


# ann = BootlegAnnotator(model_name='bootleg_hebrew', device=-1, return_embs=False, verbose=False)

import stanza
# stanza.download('he')
# import spacy_stanza
# nlp = spacy_stanza.load_pipeline('he', processors='tokenize,lemma,mwt,pos', verbose=True, use_gpu=False)

nlp = stanza.Pipeline('he', processors='tokenize,lemma,mwt,pos', verbose=True, use_gpu=False)

# import spacy_udpipe
# spacy_udpipe.download('he')
# nlp = spacy_udpipe.load('he')

test_sentences = '''בפני אלף צופים באצטדיון שוני, הזמר סיפר כי הוא ובעלו, חבר הכנסת עידן רול, מרחיבים את המשפחה ובעוד כמה חודשים תצטרף תינוקת למשפחה, אחות קטנה לארי. צפו ברגע המרגש לסיפור המלא...
בהיריון שלישי עם בת בבטן, מנחת הטלוויזיה מצטלמת לקמפיין אופנה ועונה לכל מי שטוען שהיא תיזמנה את תאריך החשיפה. "זה נושא אישי ושלושת החודשים הראשונים הם מצב חרדתי, המון דברים יכולים להשתבש". צפו
המלכה מציינת יום הולדת בודד במיוחד, ימים ספורים לאחר הלוויה של בעלה, הנסיך פיליפ. חברה הקרוב ומנהל האורוות שלה הלך לעולמו, בנה ונכדה לא מתוכננים להגיע לארמון והארי כבר חזר לאמריקה. לסיפור המלא...
בריאיון לקראת סרטה החדש שצפוי לצאת לאקרנים, סיפרה השחקנית ל-Entertainment Weekly על הוויתורים שנאלצה לעשות בעקבות השינוי במצב המשפחתי. "הייתי צריכה לעשות רק עבודות קצרות כדי להיות יותר בבית, זו האמת". לסיפור המלא...
הזמר ואשתו, יהודית באומן, נתפסו בעדשת הפפראצי באזור ביתם שבתל אביב. תבורי יצא ראשון מהרכב ומיד ניגש לפתוח את הדלת לאשתו, שזיהתה אותנו במהרה ושלחה לעברנו נשיקה. לסיפור המלא...
דור רפאלי: "אנשים לא יודעים עליי כלום"
פגשנו את המאורסת הטרייה באירוע השקה ושמענו ממנה על התכנונים לחתונה הקרובה עם בן הזוג איתי פז, איך מערכת היחסים שלו עם בנה, יאן, ומה הסטטוס עם האקס, אייל גולן. צפו
מצלמת הפפראצי שלנו נתקלה במגישת הטלוויזיה ובבן זוגה החדש, השף יובל בן נריה, בזמן שסעדו במסעדה בתל אביב. בן נריה נראה נבוך מעט מנוכחותנו, אך התאושש במהרה והשניים המשיכו בבילוי המשותף. לסיפור המלא...
החזרה לשגרה מביאה עמה גל גדול של אירועים והשקות, והפעם היה זה תורם של כוכבי "מקיף מילאנו", דרמת הנוער של "כאן" חינוכית, להשיק את תוכניתם החדשה. לצד אבני ואזולאי נכחו במקום גם כוכבים ותיקים כמו לירון וייסמן, משה אשכנזי ובועז קונפורטי, ואיתם השחקנים הצעירים שבקרוב יהפכו כנראה למוכרים מאוד. לסיפור המלא...
כוכבת הסדרה "90210" סיפרה כי לאורך השנים היו לה מספר זהויות שצצו בתוכה מדי פעם: "רגע אחד הייתי קשוחה עם פאה שחורה, ובפעם אחרת הייתי ילדת פרחים", אמרה. בעבר חשפה מקורד כי בילדותה עברה שרשרת תקיפות מיניות, וכי נאנסה כשהייתה בת 18. לסיפור המלא...
פרמיירת העונה התשיעית של "מחוברים" קיבצה את כוכביה: לוסי אהריש, עברי לידר, דורין אטיאס, מיכל אנסקי ודור רפאלי, ואם להאמין לדבריהן של אנסקי ואהריש - רפאלי הוא זה שהולך לגנוב את ההצגה: "הוא נכנס לי לנשמה, הוא ההפתעה של העונה". אטיאס מצידה, הצהירה: "אנשים לא יודעים עליי את האמת. הם חושבים שאני הכלבה מפינס". לסיפור המלא...
הדוכסית מקיימברידג' הצליחה לאחד בין האחים הנסיכים וויליאם והארי בסוף הלוויה של הנסיך פיליפ, וברגע אחד השכיחה את הטענות הקשות שהפנו אליה הארי ומייגן וחזרה להיות הנסיכה האהובה של העם. לסיפור המלא...'''

test_sentences = test_sentences.split('\n')

part_of_speech_map = {
    'NOUN': 'n',
    'PRON': 'p',
    'PROPN': 'r',
    'SPACE': ' ',
    'ADJ': 'a',
    'ADP': 'd',
    'ADV': 'v',
    'AUX': 'u',
    'CCONJ': 'c',
    'CONJ': 'j',
    'DET': 'e',
    'EOL': '.',
    'INTJ': 't',
    'NO_TAG': 'o',
    'NUM': '8',
    'PART': 'z',
    'PUNCT': '!',
    'SCONJ': '~',
    'SYM': '@',
    'VERB': '/',
    'X': 'x'
}
part_of_speech_map.setdefault('o')
regexs = [
    'p*e*n+e*r*',
    'r*e*n+e*p*',
    'n*e*r+e*p*',
    'p*e*r+e*n*',
    'n*e*p+e*r*',
    'r*e*p+e*n*'
]
regexs_compiled = [re.compile(rx) for rx in regexs]

class Chunk:
    def __init__(self, tokens):
        first_token = tokens[0]
        last_token = tokens[-1]
        self.start = first_token.parent.start_char
        self.end = last_token.parent.end_char
        self.tokens = tokens

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    @staticmethod
    def is_det(token):
        return token.text in ['ה', 'ה_'] and token.upos in ['DET']

    def permutations(self, start_token = 0, priors=None, space=''):
        if priors is None:
            priors = ['']
        perms = []
        token = self.tokens[start_token]
        token_is_det = Chunk.is_det(token)
        start_token += 1
        if start_token == len(self.tokens) and token_is_det:
            return list(set(priors))
        text = token.text
        if '_' not in text:
            next_space = ' ' if not token_is_det else ''
            for prior in priors:
                lemma = token.lemma
                perms.append(f'{prior}{space}{text}')
                if Chunk.is_det(token):
                    perms.append(f'{prior}{space}')
                elif lemma != text:
                    perms.append(f'{prior}{space}{lemma}')
        else:
            next_space = ''

        if start_token < len(self.tokens):
            perms = self.permutations(start_token, perms, next_space)
        return list(set(perms))

def hebrew_noun_chunks(doc):
    doc_sub_tokens = []
    for sent in doc.sentences:
        for token in sent.tokens:
            doc_sub_tokens.extend(word for word in token.words)
    pos_string = ''.join([part_of_speech_map[sub_token.upos] for sub_token in doc_sub_tokens])
    candidates = []
    for rx in regexs_compiled:
        for x in rx.finditer(pos_string, overlapped=True):
            chunk = Chunk(doc_sub_tokens[x.start():x.end()])
            # perms = chunk.permutations()
            # for perm in perms:
            #     print(f'{chunk.start}, {chunk.end}: {perm}')
            candidates.append(chunk)
    return list(set(candidates))

# for sentence in test_sentences:
#     print(f'{sentence}\n')
#     chunks = hebrew_noun_chunks(nlp(sentence))

def hebrew_labeler(sentence, all_aliases, min_alias_len=0, max_alias_len=7):
    used_aliases = []
    # Remove multiple spaces and replace with single - tokenization eats multiple spaces but
    # ngrams doesn't which can cause parse issues
    split_sent = sentence.strip().split()
    sentence = " ".join(split_sent)
    doc = nlp(sentence)
    used_aliases = []
    spans = []
    for chunk in hebrew_noun_chunks(doc):
        for perm in chunk.permutations():
            # print(f'perm: {perm}')
            if perm in all_aliases:
                # print(f'Found alias: {perm}')
                used_aliases.append(perm)
                spans.append((chunk.start, chunk.end))
    return used_aliases, spans


from bootleg.end2end.bootleg_annotator import BootlegAnnotator

# You can also pass `return_embs=True` to get the embeddings
ann = BootlegAnnotator(device=-1, return_embs=False, verbose=False, model_name='bootleg_hebrew')
all_aliases_trie = ann.all_aliases_trie

# all_of_them = all_aliases_trie.to_dict()
# for ann in all_of_them:
#     print(f'{ann}: {all_of_them[ann]}')

for sent in test_sentences[:50]:
    print(sent)
    print(json.dumps(ann.label_mentions(sent, hebrew_labeler), indent=4, ensure_ascii=False, default=str))


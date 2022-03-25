import json

import regex as re
import stanza

from bootleg.symbols.entity_profile import EntityProfile

stanza.download('he')
nlp = stanza.Pipeline('he', processors='tokenize,lemma,mwt,pos', verbose=True, use_gpu=True)

test_sentences = '''בפני אלף צופים באצטדיון שוני, הזמר סיפר כי הוא ובעלו, חבר הכנסת עידן רול, מרחיבים את המשפחה ובעוד כמה חודשים תצטרף תינוקת למשפחה, אחות קטנה לארי. צפו ברגע המרגש לסיפור המלא...
בהיריון שלישי עם בת בבטן, מנחת הטלוויזיה מצטלמת לקמפיין אופנה ועונה לכל מי שטוען שהיא תיזמנה את תאריך החשיפה. "זה נושא אישי ושלושת החודשים הראשונים הם מצב חרדתי, המון דברים יכולים להשתבש". צפו
המלכה מציינת יום הולדת בודד במיוחד, ימים ספורים לאחר הלוויה של בעלה, הנסיך פיליפ. חברה הקרוב ומנהל האורוות שלה הלך לעולמו, בנה ונכדה לא מתוכננים להגיע לארמון והארי כבר חזר לאמריקה. לסיפור המלא...
בריאיון לקראת סרטה החדש שצפוי לצאת לאקרנים, סיפרה השחקנית ל-Entertainment Weekly על הוויתורים שנאלצה לעשות בעקבות השינוי במצב המשפחתי. "הייתי צריכה לעשות רק עבודות קצרות כדי להיות יותר בבית, זו האמת". לסיפור המלא...
הזמר ואשתו, יהודית באומן, נתפסו בעדשת הפפראצי באזור ביתם שבתל אביב. תבורי יצא ראשון מהרכב ומיד ניגש לפתוח את הדלת לאשתו, שזיהתה אותנו במהרה ושלחה לעברנו נשיקה. לסיפור המלא...
'''

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
    def __init__(self, tokens, start_token, end_token):
        self.start_token = start_token
        self.end_token = end_token
        self.start = tokens[0].parent.start_char
        self.end = tokens[-1].parent.end_char
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
            chunk = Chunk(doc_sub_tokens[x.start():x.end()], x.start(), x.end())
            # perms = chunk.permutations()
            # for perm in perms:
            #     print(f'{chunk.start}, {chunk.end}: {perm}')
            candidates.append(chunk)
    return list(set(candidates))

# for sentence in test_sentences:
#     print(f'{sentence}\n')
#     chunks = hebrew_noun_chunks(nlp(sentence))

def hebrew_labeler(sentence, all_aliases, min_alias_len=0, max_alias_len=7):
    # Remove multiple spaces and replace with single - tokenization eats multiple spaces but
    # ngrams doesn't which can cause parse issues
    split_sent = sentence.strip().split()
    sentence = " ".join(split_sent)
    doc = nlp(sentence)
    used_aliases = []
    spans = []
    char_spans = []
    for chunk in hebrew_noun_chunks(doc):
        for perm in chunk.permutations():
            # print(f'perm: {perm}')
            if perm in all_aliases:
                # print(f'Found alias: {perm}')
                used_aliases.append(perm)
                spans.append((chunk.start_token, chunk.end_token))
                char_spans.append((chunk.start, chunk.end))
    return used_aliases, spans, char_spans


from bootleg.end2end.bootleg_annotator import BootlegAnnotator

# db = EntityProfile.load_from_cache("/home/rubmz/.cache/torch/bootleg/data/entity_db")
# db.save("/home/rubmz/.cache/torch/bootleg/data/entity_db")

# You can also pass `return_embs=True` to get the embeddings
ann = BootlegAnnotator(device=0, return_embs=False, verbose=False, model_name='bootleg_hebrew')
all_aliases_trie = ann.all_aliases_trie

# all_of_them = all_aliases_trie.to_dict()
# for ann in all_of_them:
#     print(f'{ann}: {all_of_them[ann]}')

for sent in test_sentences[1:50]:
    print(sent)
    print(json.dumps(ann.label_mentions(sent, hebrew_labeler), indent=4, ensure_ascii=False, default=str))


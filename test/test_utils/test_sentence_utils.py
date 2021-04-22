import unittest

from nltk import SpaceTokenizer
from transformers import BertTokenizer

from bootleg.utils.sentence_utils import split_sentence


def load_tokenizer(is_bert):
    if is_bert:
        return BertTokenizer.from_pretrained(
            "bert-base-cased",
            do_lower_case=False,
            cache_dir="test/data/emb_data/pretrained_bert_models",
            use_fast=False,
        )
    else:
        return SpaceTokenizer()


class SentenceUtils(unittest.TestCase):
    def test_split_sentence1(self):
        is_bert = False
        tokenizer = load_tokenizer(is_bert)
        # Test if the sentence splits correctly when the sentence meets all limits (max_aliases, sequence_length)
        max_aliases = 30
        max_seq_len = 24

        # Manually created data
        sentence = (
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because we want "
            "our cat and our alias5"
        )
        aliases = ["The big", "alias3", "alias5"]
        aliases_to_predict = [0, 1, 2]
        spans = [[0, 2], [12, 13], [20, 21]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth data
        true_phrase_arr = [
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because we want our "
            "cat and our alias5 <pad> <pad> <pad>".split(" ")
        ]
        true_spans_arr = [[[0, 2], [12, 13], [20, 21]]]
        true_alias_to_predict_arr = [[0, 1, 2]]
        true_phrase_token_pos_arr = [list(range(21)) + [-1, -1, -1]]
        true_aliases_arr = [["The big", "alias3", "alias5"]]

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_split_sentence_max_aliases(self):
        is_bert = False
        tokenizer = load_tokenizer(is_bert)
        # Test if the sentence splits correctly when max_aliases is less than the number of aliases
        max_aliases = 2
        max_seq_len = 24

        # Manually created data
        sentence = (
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because "
            "we want our cat and our alias5"
        )
        aliases = ["The big", "alias3", "alias5"]
        aliases_to_predict = [0, 1, 2]
        spans = [[0, 2], [12, 13], [20, 21]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because we want our "
            "cat and our alias5 <pad> <pad> <pad>".split(" ")
        ] * 2
        true_spans_arr = [[[0, 2], [12, 13]], [[20, 21]]]
        true_alias_to_predict_arr = [[0, 1], [0]]
        true_phrase_token_pos_arr = [list(range(21)) + [-1, -1, -1]] * 2
        true_aliases_arr = [["The big", "alias3"], ["alias5"]]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_split_sentence_alias_to_predict(self):
        is_bert = False
        tokenizer = load_tokenizer(is_bert)
        # No splitting nut change in aliases to predict...nothing should change
        max_aliases = 30
        max_seq_len = 24

        # Manually created data
        sentence = (
            "The big alias1 ran away from dogs and multi word alias2 and alias3 "
            "because we want our cat and our alias5"
        )
        aliases = ["The big", "alias3", "alias5"]
        aliases_to_predict = [0, 1]
        spans = [[0, 2], [12, 13], [20, 21]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth data
        true_phrase_arr = [
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because we want our cat "
            "and our alias5 <pad> <pad> <pad>".split(" ")
        ]
        true_spans_arr = [[[0, 2], [12, 13], [20, 21]]]
        true_alias_to_predict_arr = [[0, 1]]
        true_phrase_token_pos_arr = [list(range(21)) + [-1, -1, -1]]
        true_aliases_arr = [["The big", "alias3", "alias5"]]

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_seq_length(self):
        is_bert = False
        tokenizer = load_tokenizer(is_bert)
        # Test maximum sequence length
        max_aliases = 30
        max_seq_len = 12

        # Manual data
        sentence = (
            "The big alias1 ran away from dogs and multi word alias2 and alias3 because "
            "we want our cat and our alias5"
        )
        aliases = ["The big", "alias3", "alias5"]
        aliases_to_predict = [0, 1, 2]
        spans = [[0, 2], [12, 13], [20, 21]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            "The big alias1 ran away from dogs and multi word alias2 and".split(),
            "word alias2 and alias3 because we want our cat and our alias5".split(),
        ]
        true_spans_arr = [[[0, 2]], [[3, 4], [11, 12]]]
        true_alias_to_predict_arr = [[0], [0, 1]]
        true_phrase_token_pos_arr = [list(range(12)), list(range(9, 21))]
        true_aliases_arr = [["The big"], ["alias3", "alias5"]]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Now test with modified aliases to perdict
        aliases_to_predict = [1, 2]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            "word alias2 and alias3 because we want our cat and our alias5".split()
        ]
        true_spans_arr = [[[3, 4], [11, 12]]]
        true_alias_to_predict_arr = [[0, 1]]
        true_phrase_token_pos_arr = [list(range(9, 21))]
        true_aliases_arr = [["alias3", "alias5"]]

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_edge_case(self):
        is_bert = False
        tokenizer = load_tokenizer(is_bert)
        # Edge-case lengths

        # Test maximum sequence length
        max_aliases = 30
        max_seq_len = 3

        # Manual data
        sentence = (
            "The big alias1 ran away from dogs and multi word alias2 and alias3 "
            "because we want our cat and our alias5"
        )
        aliases = ["The big alias1", "multi word alias2 and alias3"]
        aliases_to_predict = [0, 1]
        spans = [[0, 3], [8, 13]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        # The position of a mention is taken by first token so we expect to have 1 word on
        # each side for the second alias
        true_phrase_arr = ["The big alias1".split(), "and multi word".split()]
        true_spans_arr = [[[0, 3]], [[1, 3]]]
        true_alias_to_predict_arr = [[0], [0]]
        true_phrase_token_pos_arr = [[0, 1, 2], [7, 8, 9]]
        true_aliases_arr = [["The big alias1"], ["multi word alias2 and alias3"]]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_real_cases(self):
        # Real examples we messed up
        is_bert = False
        tokenizer = load_tokenizer(is_bert)

        # EXAMPLE 1
        max_aliases = 30
        max_seq_len = 50

        # 3114|0~*~1~*~2~*~3~*~4~*~5|mexico~*~panama~*~ecuador~*~peru~*~bolivia~*~colombia|3966054~*~22997~*~9334
        # ~*~170691~*~3462~*~5222|19:20~*~36:37~*~39:40~*~44:45~*~48:49~*~70:71|The animal is called paca in most of
        # its range but tepezcuintle original Aztec language name in most of Mexico and Central America pisquinte in
        # northern Costa Rica jaleb in the Yucatán peninsula conejo pintado in Panama guanta in Ecuador majás or
        # picuro in Peru jochi pintado in Bolivia and boruga tinajo Fauna y flora de la cuenca media del Río Lebrija
        # en Rionegro Santander Humboldt Institute or guartinaja in Colombia
        sentence = (
            "The animal is called paca in most of its range but tepezcuintle original Aztec language "
            "name in most of Mexico and Central America pisquinte in northern Costa Rica jaleb in the "
            "Yucatán peninsula conejo pintado in Panama guanta in Ecuador majás or picuro in Peru jochi "
            "pintado in Bolivia and boruga tinajo Fauna y flora de la cuenca media del Río Lebrija en "
            "Rionegro Santander Humboldt Institute or guartinaja in Colombia"
        )
        aliases = ["mexico", "panama", "ecuador", "peru", "bolivia", "colombia"]
        aliases_to_predict = [0, 1, 2, 3, 4, 5]
        spans = [[19, 20], [36, 37], [39, 40], [44, 45], [48, 49], [70, 71]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            "range but tepezcuintle original Aztec language name in most of Mexico and Central America pisquinte "
            "in northern Costa Rica jaleb in the Yucatán peninsula conejo pintado in Panama guanta in Ecuador "
            "majás or picuro in Peru jochi pintado in Bolivia and boruga tinajo Fauna "
            "y flora de la cuenca media".split(),
            "Central America pisquinte in northern Costa Rica jaleb in the Yucatán peninsula conejo pintado in "
            "Panama guanta in Ecuador majás or picuro in Peru jochi pintado in Bolivia and boruga tinajo Fauna "
            "y flora de la cuenca media del Río Lebrija en Rionegro Santander Humboldt Institute or "
            "guartinaja in Colombia".split(),
        ]
        true_spans_arr = [
            [[10, 11], [27, 28], [30, 31], [35, 36], [39, 40]],
            [[15, 16], [18, 19], [23, 24], [27, 28], [49, 50]],
        ]
        true_alias_to_predict_arr = [[0, 1, 2, 3, 4], [4]]
        true_phrase_token_pos_arr = [list(range(9, 59)), list(range(21, 71))]
        true_aliases_arr = [
            ["mexico", "panama", "ecuador", "peru", "bolivia"],
            ["panama", "ecuador", "peru", "bolivia", "colombia"],
        ]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # EXAMPLE 2
        max_aliases = 10
        max_seq_len = 50

        # 20|0~*~1~*~2~*~3~*~4~*~5~*~6~*~7~*~8~*~9~*~10~*~11~*~12~*~13~*~14~*~15~*~16~*~17~*~18~*~19~*~20|coolock
        # ~*~swords~*~darndale~*~santry~*~donnycarney~*~baldoyle~*~sutton~*~donaghmede~*~artane~*~whitehall
        # ~*~kilbarrack~*~raheny~*~clontarf~*~fairview~*~malahide~*~howth~*~marino~*~ballybough~*~north
        # strand~*~sheriff street~*~east wall|1037463~*~182210~*~8554720~*~2432965~*~7890942~*~1223621~*~1008011
        # ~*~3698049~*~1469895~*~2144656~*~3628425~*~1108214~*~1564212~*~1438118~*~944694~*~1037467~*~5745962
        # ~*~2436385~*~5310245~*~12170199~*~2814197|12:13~*~14:15~*~15:16~*~17:18~*~18:19~*~19:20~*~20:21~*~21:22
        # ~*~22:23~*~23:24~*~24:25~*~25:26~*~26:27~*~27:28~*~28:29~*~29:30~*~30:31~*~38:39~*~39:41~*~41:43~*~43:45
        # |East edition The original east edition is distributed to areas such as Coolock Kilmore Swords Darndale
        # Priorswood Santry Donnycarney Baldoyle Sutton Donaghmede Artane Whitehall Kilbarrack Raheny Clontarf
        # Fairview Malahide Howth Marino and the north east inner city Summerhill Ballybough North Strand Sheriff
        # Street East Wall
        sentence = (
            "East edition The original east edition is distributed to areas such as Coolock Kilmore "
            "Swords Darndale Priorswood Santry Donnycarney Baldoyle Sutton Donaghmede Artane Whitehall "
            "Kilbarrack Raheny Clontarf Fairview Malahide Howth Marino and the north east inner city "
            "Summerhill Ballybough North Strand Sheriff Street East Wall"
        )
        aliases = [
            "coolock",
            "swords",
            "darndale",
            "santry",
            "donnycarney",
            "baldoyle",
            "sutton",
            "donaghmede",
            "artane",
            "whitehall",
            "kilbarrack",
            "raheny",
            "clontarf",
            "fairview",
            "malahide",
            "howth",
            "marino",
            "ballybough",
            "north strand",
            "sheriff street",
            "east wall",
        ]
        aliases_to_predict = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            171,
            8,
            19,
            20,
        ]
        spans = [
            [12, 13],
            [14, 15],
            [15, 16],
            [17, 18],
            [18, 19],
            [19, 20],
            [20, 21],
            [21, 22],
            [22, 23],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
            [27, 28],
            [28, 29],
            [29, 30],
            [30, 31],
            [38, 39],
            [39, 41],
            [41, 43],
            [43, 45],
        ]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth
        true_phrase_arr = [
            "East edition The original east edition is distributed to areas such as Coolock Kilmore Swords Darndale "
            "Priorswood Santry Donnycarney Baldoyle Sutton Donaghmede Artane Whitehall Kilbarrack Raheny Clontarf "
            "Fairview Malahide Howth Marino and the north east inner city Summerhill Ballybough North Strand Sheriff "
            "Street East Wall <pad> <pad> <pad> <pad> <pad>".split(),
            "East edition The original east edition is distributed to areas such as Coolock Kilmore Swords Darndale "
            "Priorswood Santry Donnycarney Baldoyle Sutton Donaghmede Artane Whitehall Kilbarrack Raheny Clontarf "
            "Fairview Malahide Howth Marino and the north east inner city Summerhill Ballybough North Strand Sheriff "
            "Street East Wall <pad> <pad> <pad> <pad> <pad>".split(),
            "East edition The original east edition is distributed to areas such as Coolock Kilmore Swords Darndale "
            "Priorswood Santry Donnycarney Baldoyle Sutton Donaghmede Artane Whitehall Kilbarrack Raheny Clontarf "
            "Fairview Malahide Howth Marino and the north east inner city Summerhill Ballybough North Strand Sheriff "
            "Street East Wall <pad> <pad> <pad> <pad> <pad>".split(),
        ]
        true_spans_arr = [
            [
                [12, 13],
                [14, 15],
                [15, 16],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [21, 22],
            ],
            [
                [20, 21],
                [21, 22],
                [22, 23],
                [23, 24],
                [24, 25],
                [25, 26],
                [26, 27],
                [27, 28],
                [28, 29],
            ],
            [
                [27, 28],
                [28, 29],
                [29, 30],
                [30, 31],
                [38, 39],
                [39, 41],
                [41, 43],
                [43, 45],
            ],
        ]
        true_alias_to_predict_arr = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 6, 7],
        ]
        true_phrase_token_pos_arr = [
            list(range(45)) + [-1] * 5,
            list(range(45)) + [-1] * 5,
            list(range(45)) + [-1] * 5,
        ]
        true_aliases_arr = [
            [
                "coolock",
                "swords",
                "darndale",
                "santry",
                "donnycarney",
                "baldoyle",
                "sutton",
                "donaghmede",
            ],
            [
                "sutton",
                "donaghmede",
                "artane",
                "whitehall",
                "kilbarrack",
                "raheny",
                "clontarf",
                "fairview",
                "malahide",
            ],
            [
                "fairview",
                "malahide",
                "howth",
                "marino",
                "ballybough",
                "north strand",
                "sheriff street",
                "east wall",
            ],
        ]

        assert len(idxs_arr) == 3
        assert len(aliases_to_predict_arr) == 3
        assert len(spans_arr) == 3
        assert len(phrase_tokens_arr) == 3
        assert len(phrase_tokens_pos_idxs_arr) == 3
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Example 2
        max_aliases = 10
        max_seq_len = 100

        # 84|0~*~1|kentucky~*~green|621151~*~478999|8:9~*~9:10|The Assembly also reserved tolls collected on the
        # Kentucky Green and Barren rivers for education and passed a two percent property tax to fund the state s
        # schools
        sentence = (
            "The Assembly also reserved tolls collected on the Kentucky Green and Barren rivers for "
            "education and passed a two percent property tax to fund the state s schools"
        )
        aliases = ["kentucky", "green"]
        aliases_to_predict = [0, 1]
        spans = [[8, 9], [9, 10]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            "The Assembly also reserved tolls collected on the Kentucky Green and Barren rivers for education and "
            "passed a two percent property tax to fund the state s schools <pad> <pad> <pad> <pad> <pad> <pad> "
            "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
            "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
            "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
            "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>".split()
        ]
        true_spans_arr = [[[8, 9], [9, 10]]]
        true_alias_to_predict_arr = [[0, 1]]
        true_phrase_token_pos_arr = [list(range(28)) + [-1] * 72]
        true_aliases_arr = [["kentucky", "green"]]

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_split_sentence_bert(self):
        is_bert = True
        tokenizer = load_tokenizer(is_bert)

        # Example 1
        max_aliases = 30
        max_seq_len = 22

        # Manual data
        sentence = "Kittens love purpleish pupppeteers because alias2 and spanning the brreaches alias5"
        aliases = ["Kittens love", "alias2", "alias5"]
        spans = [[0, 2], [5, 6], [10, 11]]
        aliases_to_predict = [0, 1, 2]

        # Truth
        bert_tokenized = [
            "Kit",
            "##tens",
            "love",
            "purple",
            "##ish",
            "pu",
            "##pp",
            "##pet",
            "##eers",
            "because",
            "alias",
            "##2",
            "and",
            "spanning",
            "the",
            "br",
            "##rea",
            "##ches",
            "alias",
            "##5",
        ]
        true_phrase_arr = [["[CLS]"] + bert_tokenized + ["[SEP]"]]
        true_spans_arr = [[[1, 4], [11, 13], [19, 21]]]
        true_alias_to_predict_arr = [[0, 1, 2]]
        true_phrase_token_pos_arr = [[-2] + list(range(20)) + [-3]]
        true_aliases_arr = [["Kittens love", "alias2", "alias5"]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Example 2
        max_aliases = 30
        max_seq_len = 9

        # Manual data
        sentence = "Kittens love purpleish pupppeteers because alias2 and spanning the brreaches alias5"
        aliases = ["Kittens love", "alias2", "alias5"]
        spans = [[0, 2], [5, 6], [10, 11]]
        aliases_to_predict = [0, 1, 2]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "Kit",
                "##tens",
                "love",
                "purple",
                "##ish",
                "pu",
                "##pp",
                "[SEP]",
            ],
            [
                "[CLS]",
                "##pet",
                "##eers",
                "because",
                "alias",
                "##2",
                "and",
                "spanning",
                "[SEP]",
            ],
            [
                "[CLS]",
                "spanning",
                "the",
                "br",
                "##rea",
                "##ches",
                "alias",
                "##5",
                "[SEP]",
            ],
        ]
        true_spans_arr = [[[1, 4]], [[4, 6]], [[6, 8]]]
        true_alias_to_predict_arr = [[0], [0], [0]]
        true_phrase_token_pos_arr = [
            [-2] + list(range(7)) + [-3],
            [-2] + list(range(7, 14)) + [-3],
            [-2] + list(range(13, 20)) + [-3],
        ]
        true_aliases_arr = [["Kittens love"], ["alias2"], ["alias5"]]

        assert len(idxs_arr) == 3
        assert len(aliases_to_predict_arr) == 3
        assert len(spans_arr) == 3
        assert len(phrase_tokens_arr) == 3
        assert len(phrase_tokens_pos_idxs_arr) == 3
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Example 3: Test greedy nature of algorithm. It will greedily pack the first two aliases together and the
        # last alias will be split up even though the second alias is also in the second split.
        max_aliases = 30
        max_seq_len = 20

        # Manual data
        sentence = (
            "Kittens Kittens Kittens Kittens love purpleish pupppeteers because alias2 and "
            "spanning the brreaches alias5"
        )
        aliases = ["Kittens love", "alias2", "alias5"]
        spans = [[3, 5], [8, 9], [13, 14]]
        aliases_to_predict = [0, 1, 2]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # True data
        true_phrase_arr = [
            [
                "[CLS]",
                "##tens",
                "Kit",
                "##tens",
                "Kit",
                "##tens",
                "love",
                "purple",
                "##ish",
                "pu",
                "##pp",
                "##pet",
                "##eers",
                "because",
                "alias",
                "##2",
                "and",
                "spanning",
                "the",
                "[SEP]",
            ],
            [
                "[CLS]",
                "love",
                "purple",
                "##ish",
                "pu",
                "##pp",
                "##pet",
                "##eers",
                "because",
                "alias",
                "##2",
                "and",
                "spanning",
                "the",
                "br",
                "##rea",
                "##ches",
                "alias",
                "##5",
                "[SEP]",
            ],
        ]
        true_spans_arr = [[[4, 7], [14, 16]], [[9, 11], [17, 19]]]
        true_alias_to_predict_arr = [[0, 1], [1]]
        true_phrase_token_pos_arr = [
            [-2] + list(range(3, 21)) + [-3],
            [-2] + list(range(8, 26)) + [-3],
        ]
        true_aliases_arr = [["Kittens love", "alias2"], ["alias2", "alias5"]]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Example 4: testing if alias goes off the edge
        max_aliases = 4
        max_seq_len = 7

        # Manual data
        sentence = "alias1 or multi word alias2"
        aliases = ["alias1", "multi word alias2"]
        spans = [[0, 1], [2, 5]]
        aliases_to_predict = [0, 1]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth
        true_phrase_arr = [["[CLS]", "alias", "##1", "or", "multi", "word", "[SEP]"]]
        true_spans_arr = [[[1, 3], [4, 6]]]
        true_alias_to_predict_arr = [[0, 1]]
        true_phrase_token_pos_arr = [[-2] + list(range(5)) + [-3]]
        true_aliases_arr = [["alias1", "multi word alias2"]]

        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # Test if the sentence splits correctly when the aliases span boundaries on both ends
        max_aliases = 2
        max_seq_len = 7

        # Manually created data
        sentence = "alias3 alias4 alias3"
        aliases = ["alias3", "alias4", "alias3"]
        aliases_to_predict = [0, 1, 2]
        spans = [[0, 1], [1, 2], [2, 3]]

        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )

        # Truth data
        true_phrase_arr = [
            "[CLS] alias ##3 alias ##4 alias [SEP]".split(" "),
            "[CLS] ##3 alias ##4 alias ##3 [SEP]".split(" "),
        ]
        true_spans_arr = [[[1, 3], [3, 5]], [[2, 4], [4, 6]]]
        true_alias_to_predict_arr = [[0, 1], [1]]
        true_phrase_token_pos_arr = [
            [-2] + list(range(5)) + [-3],
            [-2] + [1, 2, 3, 4, 5] + [-3],
        ]
        true_aliases_arr = [["alias3", "alias4"], ["alias4", "alias3"]]

        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

    def test_real_cases_bert(self):
        # Example 1
        max_aliases = 10
        max_seq_len = 102
        is_bert = True
        tokenizer = load_tokenizer(is_bert)

        # Manual data
        sentence = (
            "The guest roster for O'Brien 's final show on January 22\u2014 Tom Hanks , Steve Carell and "
            "original first guest Will Ferrell \u2014was regarded by O'Brien as a `` dream lineup '' ; "
            "in addition , Neil Young performed his song `` Long May You Run `` and , as the show closed , "
            "was joined by Beck , Ferrell ( dressed as Ronnie Van Zant ) , Billy Gibbons , Ben Harper , "
            "O'Brien , Viveca Paulin , and The Tonight Show Band to perform the Lynyrd Skynyrd song `` "
            "Free Bird `` ."
        )
        aliases = [
            "tom hanks",
            "steve carell",
            "will ferrell",
            "neil young",
            "long may you run",
            "beck",
            "ronnie van zant",
            "billy gibbons",
            "ben harper",
            "viveca paulin",
            "lynyrd skynyrd",
            "free bird",
        ]
        spans = [
            [11, 13],
            [14, 16],
            [20, 22],
            [36, 38],
            [42, 46],
            [57, 58],
            [63, 66],
            [68, 70],
            [71, 73],
            [76, 78],
            [87, 89],
            [91, 93],
        ]
        aliases_to_predict = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "The",
                "guest",
                "roster",
                "for",
                "O",
                "'",
                "Brien",
                "'",
                "s",
                "final",
                "show",
                "on",
                "January",
                "22",
                "—",
                "Tom",
                "Hank",
                "##s",
                ",",
                "Steve",
                "Care",
                "##ll",
                "and",
                "original",
                "first",
                "guest",
                "Will",
                "Fe",
                "##rrell",
                "—",
                "was",
                "regarded",
                "by",
                "O",
                "'",
                "Brien",
                "as",
                "a",
                "`",
                "`",
                "dream",
                "lineup",
                "'",
                "'",
                ";",
                "in",
                "addition",
                ",",
                "Neil",
                "Young",
                "performed",
                "his",
                "song",
                "`",
                "`",
                "Long",
                "May",
                "You",
                "Run",
                "`",
                "`",
                "and",
                ",",
                "as",
                "the",
                "show",
                "closed",
                ",",
                "was",
                "joined",
                "by",
                "Beck",
                ",",
                "Fe",
                "##rrell",
                "(",
                "dressed",
                "as",
                "Ronnie",
                "Van",
                "Z",
                "##ant",
                ")",
                ",",
                "Billy",
                "Gibbons",
                ",",
                "Ben",
                "Harper",
                ",",
                "O",
                "'",
                "Brien",
                ",",
                "V",
                "##ive",
                "##ca",
                "Paul",
                "##in",
                ",",
                "[SEP]",
            ],
            [
                "[CLS]",
                "original",
                "first",
                "guest",
                "Will",
                "Fe",
                "##rrell",
                "—",
                "was",
                "regarded",
                "by",
                "O",
                "'",
                "Brien",
                "as",
                "a",
                "`",
                "`",
                "dream",
                "lineup",
                "'",
                "'",
                ";",
                "in",
                "addition",
                ",",
                "Neil",
                "Young",
                "performed",
                "his",
                "song",
                "`",
                "`",
                "Long",
                "May",
                "You",
                "Run",
                "`",
                "`",
                "and",
                ",",
                "as",
                "the",
                "show",
                "closed",
                ",",
                "was",
                "joined",
                "by",
                "Beck",
                ",",
                "Fe",
                "##rrell",
                "(",
                "dressed",
                "as",
                "Ronnie",
                "Van",
                "Z",
                "##ant",
                ")",
                ",",
                "Billy",
                "Gibbons",
                ",",
                "Ben",
                "Harper",
                ",",
                "O",
                "'",
                "Brien",
                ",",
                "V",
                "##ive",
                "##ca",
                "Paul",
                "##in",
                ",",
                "and",
                "The",
                "Tonight",
                "Show",
                "Band",
                "to",
                "perform",
                "the",
                "L",
                "##yn",
                "##yr",
                "##d",
                "Sky",
                "##ny",
                "##rd",
                "song",
                "`",
                "`",
                "Free",
                "Bird",
                "`",
                "`",
                ".",
                "[SEP]",
            ],
        ]
        true_spans_arr = [
            [
                [16, 19],
                [20, 23],
                [27, 30],
                [49, 51],
                [56, 60],
                [72, 73],
                [79, 83],
                [85, 87],
                [88, 90],
                [95, 100],
            ],
            [
                [4, 7],
                [26, 28],
                [33, 37],
                [49, 50],
                [56, 60],
                [62, 64],
                [65, 67],
                [72, 77],
                [86, 93],
                [96, 98],
            ],
        ]
        true_alias_to_predict_arr = [[0, 1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9]]
        true_phrase_token_pos_arr = [
            [-2] + list(range(100)) + [-3],
            [-2] + list(range(23, 123)) + [-3],
        ]
        true_aliases_arr = [
            [
                "tom hanks",
                "steve carell",
                "will ferrell",
                "neil young",
                "long may you run",
                "beck",
                "ronnie van zant",
                "billy gibbons",
                "ben harper",
                "viveca paulin",
            ],
            [
                "will ferrell",
                "neil young",
                "long may you run",
                "beck",
                "ronnie van zant",
                "billy gibbons",
                "ben harper",
                "viveca paulin",
                "lynyrd skynyrd",
                "free bird",
            ],
        ]
        # Run function
        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 2
        assert len(aliases_to_predict_arr) == 2
        assert len(spans_arr) == 2
        assert len(phrase_tokens_arr) == 2
        assert len(true_phrase_arr) == 2
        assert len(phrase_tokens_pos_idxs_arr) == 2

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        max_aliases = 10
        max_seq_len = 102

        # Manual data sentence = "Alexander Rae Baldwin III ( born April 3 , 1958 , in Massapequa , Long Island ,
        # New York , USA ) is an American actor who is the oldest and best known of the \" Baldwin brothers \" ,
        # with brothers Daniel , Stephen and William ." aliases = ["april 3", "other events of 1958", "massapequa",
        # "long island", "united states", "actor", "baldwin brothers", "leroy", "stephen baldwin",
        # "william baldwin"] spans = [[6, 8], [9, 10], [12, 13], [14, 16], [20, 21], [25, 26], [36, 38], [42, 43],
        # [44, 45], [46, 47]] aliases_to_predict = [0,1,2,3,4,5,6,7,8,9]

        sentence = (
            "Alexander få Baldwin III (born April 3, 1958, in Massapequa, Long Island, New York, USA) is an "
            'American actor who is the oldest and best known of the "Baldwin brothers", with brothers '
            "Daniel, Stephen and William."
        )
        aliases = [
            "april 3",
            "other events of 1958",
            "massapequa",
            "long island",
            "united states",
            "actor",
            "baldwin brothers",
            "leroy",
            "stephen baldwin",
            "william baldwin",
        ]
        spans = [
            [5, 7],
            [7, 8],
            [9, 10],
            [10, 12],
            [14, 15],
            [18, 19],
            [28, 30],
            [32, 33],
            [33, 34],
            [35, 36],
        ]
        aliases_to_predict = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "Alexander",
                "f",
                "##å",
                "Baldwin",
                "III",
                "(",
                "born",
                "April",
                "3",
                ",",
                "1958",
                ",",
                "in",
                "Mass",
                "##ap",
                "##e",
                "##qua",
                ",",
                "Long",
                "Island",
                ",",
                "New",
                "York",
                ",",
                "USA",
                ")",
                "is",
                "an",
                "American",
                "actor",
                "who",
                "is",
                "the",
                "oldest",
                "and",
                "best",
                "known",
                "of",
                "the",
                '"',
                "Baldwin",
                "brothers",
                '"',
                ",",
                "with",
                "brothers",
                "Daniel",
                ",",
                "Stephen",
                "and",
                "William",
                ".",
                "[SEP]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
            ]
        ]
        true_spans_arr = [
            [
                [8, 11],
                [11, 13],
                [14, 19],
                [19, 22],
                [25, 27],
                [30, 31],
                [40, 45],
                [47, 49],
                [49, 50],
                [51, 53],
            ]
        ]
        true_alias_to_predict_arr = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        true_phrase_token_pos_arr = [[-2] + list(range(52)) + [-3] + [-1] * 48]
        true_aliases_arr = [
            [
                "april 3",
                "other events of 1958",
                "massapequa",
                "long island",
                "united states",
                "actor",
                "baldwin brothers",
                "leroy",
                "stephen baldwin",
                "william baldwin",
            ]
        ]
        # Run function

        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        max_aliases = 10
        max_seq_len = 102

        sentence = "It is organized by Central Research Institute of Iron and Steel of China ( 中国钢铁研究总院 ) ."
        aliases = ["china"]
        spans = [[12, 13]]
        aliases_to_predict = [0]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "It",
                "is",
                "organized",
                "by",
                "Central",
                "Research",
                "Institute",
                "of",
                "Iron",
                "and",
                "Steel",
                "of",
                "China",
                "(",
                "中",
                "国",
                "[UNK]",
                "[UNK]",
                "[UNK]",
                "[UNK]",
                "[UNK]",
                "[UNK]",
                ")",
                ".",
                "[SEP]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
            ]
        ]
        true_spans_arr = [[[13, 14]]]
        true_alias_to_predict_arr = [[0]]
        true_phrase_token_pos_arr = [[-2] + list(range(24)) + [-3] + [-1] * 76]
        true_aliases_arr = [["china"]]
        # Run function

        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        max_aliases = 10
        max_seq_len = 102
        # There is a special unicode character at , ‍‍‍‍ '' Upal (in between ,  and '' of \u200d\u200d\u200d\u200d -
        # this gets cleaned
        sentence = (
            "Upal ( , Wade-Giles : Wup ‘ aêrh Hsiang , Xiao'erjing : ءُپَاعَر سِيْا , ‍‍‍‍ '' Upal '' , Упал ) is a "
            "small town in western Xinjiang , China ."
        )
        aliases = ["upal", "xiaoerjing", "upal", "xinjiang", "china"]
        spans = [[0, 1], [10, 11], [15, 18], [28, 29], [30, 31]]
        aliases_to_predict = [0, 1, 2, 3, 4]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "Up",
                "##al",
                "(",
                ",",
                "Wade",
                "-",
                "Giles",
                ":",
                "Wu",
                "##p",
                "‘",
                "a",
                "##ê",
                "##r",
                "##h",
                "H",
                "##sian",
                "##g",
                ",",
                "Xiao",
                "'",
                "er",
                "##jin",
                "##g",
                ":",
                "[UNK]",
                "[UNK]",
                ",",
                "'",
                "'",
                "Up",
                "##al",
                "'",
                "'",
                ",",
                "У",
                "##п",
                "##а",
                "##л",
                ")",
                "is",
                "a",
                "small",
                "town",
                "in",
                "western",
                "Xi",
                "##nji",
                "##ang",
                ",",
                "China",
                ".",
                "[SEP]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
            ]
        ]
        true_spans_arr = [[[1, 3], [20, 25], [29, 33], [47, 50], [51, 52]]]
        true_alias_to_predict_arr = [[0, 1, 2, 3, 4]]
        true_phrase_token_pos_arr = [[-2] + list(range(52)) + [-3] + [-1] * 48]
        true_aliases_arr = [["upal", "xiaoerjing", "upal", "xinjiang", "china"]]
        # Run function

        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1
        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # TESTING THE ADJUSTING INDEXES WHEN ALIAS IS UNICODE CHARACTERS One of the aliases is around a unicode
        # character. We adjust the spans to drop this and take the next character instead. If we can't do this we
        # take the former character (i.e., at the end of the sentence). While we could probably do something
        # fancier, this is a rare case.

        # This is testing if we DO have a character after the unicode
        max_aliases = 10
        max_seq_len = 102
        # There is a special unicode character at , ‍‍‍‍ '' (in between ,  and '' of \u200d\u200d\u200d\u200d - this
        # gets cleaned
        sentence = "Upal Xiao , ‍‍‍‍ '"
        aliases = ["upal", "xiaoerjing", "upal"]
        spans = [[0, 1], [1, 2], [3, 4]]
        aliases_to_predict = [0, 1, 2]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "Up",
                "##al",
                "Xiao",
                ",",
                "'",
                "[SEP]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
            ]
        ]
        true_spans_arr = [[[1, 3], [3, 4], [5, 6]]]
        true_alias_to_predict_arr = [[0, 1, 2]]
        true_phrase_token_pos_arr = [[-2] + list(range(5)) + [-3] + [-1] * 95]
        true_aliases_arr = [["upal", "xiaoerjing", "upal"]]
        # Run function

        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])

        # This is testing is we do NOT have a character after the unicode
        max_aliases = 10
        max_seq_len = 102
        # There is a special unicode character at , ‍‍‍‍ '' (in between ,  and '' of \u200d\u200d\u200d\u200d -
        # this gets cleaned
        sentence = "Upal Xiao , ‍‍‍‍ "
        aliases = ["upal", "xiaoerjing", "upal"]
        spans = [[0, 1], [1, 2], [3, 4]]
        aliases_to_predict = [0, 1, 2]

        # Truth
        true_phrase_arr = [
            [
                "[CLS]",
                "Up",
                "##al",
                "Xiao",
                ",",
                "[SEP]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
                "[PAD]",
            ]
        ]
        true_spans_arr = [[[1, 3], [3, 4], [4, 5]]]
        true_alias_to_predict_arr = [[0, 1, 2]]
        true_phrase_token_pos_arr = [[-2] + list(range(4)) + [-3] + [-1] * 96]
        true_aliases_arr = [["upal", "xiaoerjing", "upal"]]
        # Run function

        (
            idxs_arr,
            aliases_to_predict_arr,
            spans_arr,
            phrase_tokens_arr,
            phrase_tokens_pos_idxs_arr,
        ) = split_sentence(
            max_aliases,
            sentence,
            spans,
            aliases,
            aliases_to_predict,
            max_seq_len,
            is_bert,
            tokenizer,
            sanity_check=True,
        )
        assert len(idxs_arr) == 1
        assert len(aliases_to_predict_arr) == 1
        assert len(spans_arr) == 1
        assert len(phrase_tokens_arr) == 1
        assert len(phrase_tokens_pos_idxs_arr) == 1

        for i in range(len(idxs_arr)):
            self.assertEqual(len(phrase_tokens_arr[i]), max_seq_len)
            self.assertEqual(phrase_tokens_arr[i], true_phrase_arr[i])
            self.assertEqual(
                phrase_tokens_pos_idxs_arr[i], true_phrase_token_pos_arr[i]
            )
            self.assertEqual(spans_arr[i], true_spans_arr[i])
            self.assertEqual(aliases_to_predict_arr[i], true_alias_to_predict_arr[i])
            self.assertEqual([aliases[idx] for idx in idxs_arr[i]], true_aliases_arr[i])


if __name__ == "__main__":
    unittest.main()

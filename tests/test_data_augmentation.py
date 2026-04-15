import random

from codllm.data.augmentation import (
    accent_random_vowel,
    delete_random_char,
    expand_special_digraphs,
    insert_random_whitespace,
    qwerty_misspell,
    swap_adjacent_chars,
)


class TestSwapAdjacentChars:
    def test_basic_swap(self):
        random.seed(0)
        result = swap_adjacent_chars("abcd")
        assert len(result) == 4
        assert sorted(result) == sorted("abcd")

    def test_single_char_unchanged(self):
        assert swap_adjacent_chars("a") == "a"

    def test_empty_string_unchanged(self):
        assert swap_adjacent_chars("") == ""

    def test_two_chars_always_swaps(self):
        result = swap_adjacent_chars("ab")
        assert result == "ba"

    def test_preserves_length(self):
        text = "hello world"
        result = swap_adjacent_chars(text)
        assert len(result) == len(text)


class TestDeleteRandomChar:
    def test_basic_deletion(self):
        result = delete_random_char("abc")
        assert len(result) == 2

    def test_empty_string_unchanged(self):
        assert delete_random_char("") == ""

    def test_single_char_returns_empty(self):
        assert delete_random_char("x") == ""

    def test_result_is_substring(self):
        text = "hello"
        result = delete_random_char(text)
        assert len(result) == len(text) - 1
        assert all(c in text for c in result)


class TestInsertRandomWhitespace:
    def test_basic_insertion(self):
        result = insert_random_whitespace("abc")
        assert len(result) == 4
        assert " " in result

    def test_empty_string_gets_space(self):
        assert insert_random_whitespace("") == " "

    def test_adds_exactly_one_space(self):
        text = "nospaces"
        result = insert_random_whitespace(text)
        assert result.count(" ") == 1


class TestAccentRandomVowel:
    def test_no_vowels_unchanged(self):
        assert accent_random_vowel("bcdfg") == "bcdfg"

    def test_empty_string_unchanged(self):
        assert accent_random_vowel("") == ""

    def test_preserves_length(self):
        result = accent_random_vowel("hello")
        assert len(result) == len("hello")

    def test_accents_a_vowel(self):
        random.seed(42)
        result = accent_random_vowel("a")
        assert result in ["à", "á", "â", "ä"]

    def test_only_lowercase_vowels_affected(self):
        result = accent_random_vowel("ABCDE")
        assert result == "ABCDE"


class TestQwertyMisspell:
    def test_basic_misspell(self):
        random.seed(0)
        result = qwerty_misspell("hello")
        assert len(result) == len("hello")
        assert result != "hello"

    def test_empty_string_unchanged(self):
        assert qwerty_misspell("") == ""

    def test_no_letters_unchanged(self):
        assert qwerty_misspell("123!@#") == "123!@#"

    def test_preserves_case(self):
        random.seed(1)
        result = qwerty_misspell("A")
        assert result.isupper()

    def test_replacement_is_neighbor(self):
        neighbors = {
            "q": "wa",
            "w": "qeas",
            "e": "wrds",
            "r": "etdf",
            "t": "ryfg",
            "y": "tugh",
            "u": "yijh",
            "i": "uojk",
            "o": "iplk",
            "p": "ol",
            "a": "qwsz",
            "s": "awedxz",
            "d": "serfcx",
            "f": "drtgvc",
            "g": "ftyhbv",
            "h": "gyujnb",
            "j": "huiknm",
            "k": "jiolm",
            "l": "kop",
            "z": "asx",
            "x": "zsdc",
            "c": "xdfv",
            "v": "cfgb",
            "b": "vghn",
            "n": "bhjm",
            "m": "njk",
        }
        for _ in range(50):
            result = qwerty_misspell("a")
            assert result in neighbors["a"]

    def test_single_char(self):
        result = qwerty_misspell("k")
        assert result in "jiolm"


class TestExpandSpecialDigraphs:
    def test_nordic(self):
        assert expand_special_digraphs("æ") == "ae"
        assert expand_special_digraphs("ø") == "oe"
        assert expand_special_digraphs("å") == "aa"

    def test_german(self):
        assert expand_special_digraphs("ä") == "ae"
        assert expand_special_digraphs("ö") == "oe"
        assert expand_special_digraphs("ü") == "ue"
        assert expand_special_digraphs("ß") == "ss"

    def test_french_ligature(self):
        assert expand_special_digraphs("œ") == "oe"

    def test_no_special_chars_unchanged(self):
        assert expand_special_digraphs("hello") == "hello"

    def test_empty_string(self):
        assert expand_special_digraphs("") == ""

    def test_mixed_text(self):
        assert expand_special_digraphs("hætte") == "haette"

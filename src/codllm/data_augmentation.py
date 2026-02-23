import random


def swap_adjacent_chars(text: str) -> str:
    """Swap two random adjacent characters in the string. If the string is shorter than two characters, simply return it."""
    if len(text) < 2:
        return text

    chars = list(text)
    i = random.randint(0, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def delete_random_char(text: str) -> str:
    """Delete a random character from the string. If the string is empty, simply return it."""
    if len(text) == 0:
        return text

    i = random.randint(0, len(text) - 1)
    return text[:i] + text[i + 1:]


def insert_random_whitespace(text: str) -> str:
    """Insert a whitespace at a random position in the string."""
    i = random.randint(0, len(text))
    return text[:i] + " " + text[i:]


def accent_random_vowel(text: str) -> str:
    """Replace a random vowel with an accented variant. If no vowels are found, return the string unchanged."""
    vowel_to_accented = {
        "a": ["à", "á", "â", "ä"],
        "e": ["è", "é", "ê", "ë"],
        "i": ["ì", "í", "î", "ï"],
        "o": ["ò", "ó", "ô", "ö"],
        "u": ["ù", "ú", "û", "ü"],
        "y": ["ý", "ÿ"],
    }

    vowel_indices = [i for i, c in enumerate(text) if c in vowel_to_accented]
    if not vowel_indices:
        return text

    i = random.choice(vowel_indices)
    accented = random.choice(vowel_to_accented[text[i]])
    return text[:i] + accented + text[i + 1:]


def qwerty_misspell(text: str) -> str:
    """Replace a random character with an adjacent key on the QWERTY keyboard layout."""
    qwerty_neighbors: dict[str, str] = {
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

    eligible = [i for i, c in enumerate(text) if c.lower() in qwerty_neighbors]
    if not eligible:
        return text

    i = random.choice(eligible)
    original = text[i]
    replacement = random.choice(qwerty_neighbors[original.lower()])
    if original.isupper():
        replacement = replacement.upper()
    return text[:i] + replacement + text[i + 1:]


def expand_special_digraphs(text: str) -> str:
    """
    Replace European characters that historically expand
    to multiple ASCII letters (e.g. æ -> ae).
    """

    replacements = {
        # Nordic
        "æ": "ae",
        "ø": "oe",
        "å": "aa",

        # German
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",

        # Dutch / French ligatures
        "œ": "oe",

        # # Icelandic
        # "ð": "d",
        # "þ": "th",

        # # Some Slavic cases that are commonly expanded
        # "ł": "l",  # sometimes written as l
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text
import random

VOWEL_TO_ACCENTED = {
    "a": ["à", "á", "â", "ä"],
    "e": ["è", "é", "ê", "ë"],
    "i": ["ì", "í", "î", "ï"],
    "o": ["ò", "ó", "ô", "ö"],
    "u": ["ù", "ú", "û", "ü"],
    "y": ["ý", "ÿ"],
}

QWERTY_NEIGHBORS: dict[str, str] = {
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


def swap_adjacent_chars(text: str, rng: random.Random | None = None) -> str:
    """Swap two random adjacent characters in the string."""
    if len(text) < 2:
        return text

    random_source = random if rng is None else rng
    chars = list(text)
    i = random_source.randint(0, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def delete_random_char(text: str, rng: random.Random | None = None) -> str:
    """Delete a random character from the string. If the string is empty, simply return it."""
    if len(text) == 0:
        return text

    random_source = random if rng is None else rng
    i = random_source.randint(0, len(text) - 1)
    return text[:i] + text[i + 1 :]


def insert_random_whitespace(text: str, rng: random.Random | None = None) -> str:
    """Insert a whitespace at a random position in the string."""
    random_source = random if rng is None else rng
    i = random_source.randint(0, len(text))
    return text[:i] + " " + text[i:]


def accent_random_vowel(text: str, rng: random.Random | None = None) -> str:
    """Replace a random vowel with an accented variant. If no vowels are found, return the string unchanged."""
    vowel_indices = [i for i, c in enumerate(text) if c in VOWEL_TO_ACCENTED]
    if not vowel_indices:
        return text

    random_source = random if rng is None else rng
    i = random_source.choice(vowel_indices)
    accented = random_source.choice(VOWEL_TO_ACCENTED[text[i]])
    return text[:i] + accented + text[i + 1 :]


def qwerty_misspell(text: str, rng: random.Random | None = None) -> str:
    """Replace a random character with an adjacent key on the QWERTY keyboard layout."""
    eligible = [i for i, c in enumerate(text) if c.lower() in QWERTY_NEIGHBORS]
    if not eligible:
        return text

    random_source = random if rng is None else rng
    i = random_source.choice(eligible)
    original = text[i]
    replacement = random_source.choice(QWERTY_NEIGHBORS[original.lower()])
    if original.isupper():
        replacement = replacement.upper()
    return text[:i] + replacement + text[i + 1 :]


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

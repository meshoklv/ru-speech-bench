# -*- coding: utf-8 -*-
"""Minimal Russian text normalizer for Piper (prototype).

Piper's espeak-ng front end expands a *standalone* integer into words, but it
does not handle:
  * numbers with a space as thousands separator ("15 750" -> "15", "750")
  * clock times ("14:00")
  * ordinals with a hyphenated ending ("25-го", "3-го")
  * dates ("3 сентября" must be an ordinal in genitive: "третьего сентября")

This script patches exactly those four cases with regexes on top of
`num2words`. It is a prototype, NOT a grammar engine. Known limitations:
  * no case agreement after prepositions ("с четырнадцать ноль-ноль"
    instead of "с четырнадцати ноль-ноль");
  * ordinal declension is a crude ending swap on the last word; only
    "третий" has an explicit irregular table;
  * no gender/case agreement of the noun after a numeral.
For production use a full normalizer (e.g. nemo_text_processing).

Usage:
    pip install num2words
    python3 normalize_ru.py            # runs the built-in examples
    # or: from normalize_ru import normalize; normalize("... 15 750 ...")
"""
import re

from num2words import num2words

MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря")

IRREGULAR_ORDINALS = {
    "третий": {"го": "третьего", "му": "третьему", "м": "третьем",
               "х": "третьих", "я": "третья", "е": "третье"},
}


def ordinal_with_ending(n: int, ending: str) -> str:
    """Ordinal of `n` with the case/gender ending given as a hyphen suffix.

    num2words returns the nominative masculine ("двадцать пятый"); the ending
    of the last word is swapped for the requested one.
    """
    words = num2words(n, lang="ru", to="ordinal").split(" ")
    last = words[-1]
    if last in IRREGULAR_ORDINALS:
        words[-1] = IRREGULAR_ORDINALS[last].get(ending, last)
        return " ".join(words)
    for suf in ("ий", "ый", "ой"):
        if last.endswith(suf):
            stem = last[: -len(suf)]
            break
    else:
        stem = last[:-2]
    mapping = {"го": stem + "ого", "му": stem + "ому", "м": stem + "ом",
               "х": stem + "ых", "я": stem + "ая", "е": stem + "ое"}
    words[-1] = mapping.get(ending, last)   # unknown ending: keep nominative
    return " ".join(words)


def normalize(text: str) -> str:
    out = text

    # 1) date "3 сентября" -> "третьего сентября" (ordinal, genitive)
    out = re.sub(
        r"\b(\d{1,2}) (" + "|".join(MONTHS_GEN) + r")\b",
        lambda m: f"{ordinal_with_ending(int(m.group(1)), 'го')} {m.group(2)}",
        out)

    # 2) hyphenated ordinals: "25-го", "3-го", "2-м", "5-х", ...
    out = re.sub(
        r"\b(\d{1,3})-(го|му|м|х|я|е|й)\b",
        lambda m: ordinal_with_ending(int(m.group(1)), m.group(2)),
        out)

    # 3) time "14:00" -> "четырнадцать ноль-ноль", "9:05" -> "девять пять"
    def time_repl(m):
        hours, minutes = int(m.group(1)), int(m.group(2))
        h_w = num2words(hours, lang="ru")
        if minutes == 0:
            return f"{h_w} ноль-ноль"
        return f"{h_w} {num2words(minutes, lang='ru')}"
    out = re.sub(r"\b(\d{1,2}):(\d{2})\b", time_repl, out)

    # 4) thousands separated by a space: "15 750" -> "15750"
    #    (heuristic: digit, space, exactly three digits, then a non-digit)
    out = re.sub(r"(?<=\d) (\d{3})(?!\d)", r"\1", out)

    # 5) remaining bare integers -> cardinal words
    out = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="ru"), out)
    return out


if __name__ == "__main__":
    examples = [
        "Ваш заказ номер 4521 на сумму 15 750 рублей готов к выдаче.",
        "Доставка запланирована на 3 сентября, с 14:00 до 18:00. Вам удобно?",
        "К сожалению, заказ задерживается на складе до 25-го числа.",
        "Подтвердите, пожалуйста, адрес: улица Ленина, дом 12, корпус 3, квартира 45.",
    ]
    for t in examples:
        print("BEFORE:", t)
        print("AFTER: ", normalize(t))
        print()

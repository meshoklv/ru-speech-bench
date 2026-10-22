"""The same five test phrases in two forms: plain text and text with stress marks
('+' before the stressed vowel).

The Russian F5-TTS fine-tune was trained with stress-annotated text: the v2 and
accent_tune checkpoints expect it, otherwise they place stress arbitrarily.
The accented strings are hand-annotated; check them with a native speaker or RUAccent.
"""

# (key, plain text, text with stress marks)
PHRASES = [
    ("privet",
     "Здравствуйте! Меня зовут Анна, салон «Ромашка». Звоню по поводу вашей записи.",
     "Здр+авствуйте! Мен+я зов+ут +Анна, сал+он «Ром+ашка». Звон+ю по п+оводу в+ашей з+аписи."),
    ("zakaz",
     "Ваш заказ номер четыре тысячи пятьсот двадцать один готов к выдаче, "
     "сумма пятнадцать тысяч семьсот пятьдесят рублей.",
     "Ваш зак+аз н+омер ч+етыре т+ысячи пятьс+от дв+адцать од+ин гот+ов к в+ыдаче, "
     "с+умма пятн+адцать т+ысяч семьс+от пятьдес+ят рубл+ей."),
    ("dostavka",
     "Доставка запланирована на третье сентября, с двух до шести вечера. Вам удобно?",
     "Дост+авка заплан+ирована на тр+етье сентябр+я, с двух до шест+и в+ечера. Вам уд+обно?"),
    ("zaderzhka",
     "К сожалению, заказ задерживается на складе примерно на неделю.",
     "К сожал+ению, зак+аз зад+ерживается на скл+аде прим+ерно на нед+елю."),
    ("ne-ponyala",
     "Извините, я вас не совсем поняла. Повторите, пожалуйста.",
     "Извин+ите, я вас не совс+ем понял+а. Повтор+ите, пож+алуйста."),
]

# F5 REQUIRES an exact transcript of the reference clip (put it in refs.json, see run_f5.py).
# The reference must be DENSE speech without pauses. F5 estimates the output length as
#   ref_duration * (chars_of_text / chars_of_ref_transcript)
# so a reference with long pauses stretches the synthesis several-fold
# (observed: a ~6 s phrase came out at ~18 s with a gappy reference).

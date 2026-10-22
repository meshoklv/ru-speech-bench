"""Test phrases (Russian data) shared by the Piper benchmark.

Each phrase targets one front-end weakness: plain greeting, a number with a
space as thousands separator, a date + clock times, a hyphenated ordinal, an
address with several numbers, and a stress homograph ("замок").
"""
PHRASES = {
    "privet": "Здравствуйте! Меня зовут Анна, я звоню из клиники по поводу вашей записи.",
    "zakaz": "Ваш заказ номер 4521 на сумму 15 750 рублей готов к выдаче.",
    "dostavka": "Доставка запланирована на 3 сентября, с 14:00 до 18:00. Вам удобно?",
    "shanghai": "К сожалению, заказ задерживается на складе до 25-го числа.",
    "adres": "Подтвердите, пожалуйста, адрес: улица Ленина, дом 12, корпус 3, квартира 45.",
    "zamok": "Замок на двери склада заменён",
}

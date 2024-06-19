import logging

from langdetect import detect


def print_lang_text(text: str):
    print(f"----{text}----")
    print(detect(text))


def main():
    text = "War doesn't show who's right, just who's left."
    print_lang_text(text)

    text = "Ein, zwei, drei, vier"
    print_lang_text(text)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)
    main()

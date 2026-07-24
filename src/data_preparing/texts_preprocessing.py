import re
import string
import simplemma


def prepare_main_text(title: str, body_text: str) -> str:
    title, body_text = title.strip(), body_text.strip()

    return f"{title}  {body_text}"


def preprocess_and_lemmatize_text(text: str) -> str:
    text = text.lower().strip()

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"[«»“”„“—–…®™]", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split(" ")

    lemmas = [simplemma.lemmatize(w, lang="en") for w in tokens]

    final_text = " ".join(lemmas)

    return final_text


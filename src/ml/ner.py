from transformers import pipeline


def collect_initial_ner_entities(main_ner_pipeline: pipeline, main_text: str) -> list[dict]:
    initial_ner_entities = main_ner_pipeline(main_text)
    return initial_ner_entities


def collect_ner_entities(main_ner_pipeline: pipeline, main_text: str) -> list[dict]:
    ner_entities = collect_initial_ner_entities(main_ner_pipeline, main_text)

    final_entities = []
    final_words = set()

    prev_word = None
    prev_entity_group = None
    for entity in ner_entities:
        cur_word = entity["word"]
        cur_entity_group = entity["entity_group"]
        if prev_word is None:
            prev_word = cur_word
            prev_entity_group = cur_entity_group
        else:
            if cur_word.startswith("#"):
                cur_word = cur_word.strip("#")
                prev_word += cur_word
            else:
                cur_dct = {
                    "word": prev_word,
                    "entity_group": prev_entity_group
                }
                if prev_word not in final_words:
                    final_entities.append(cur_dct)
                    final_words.add(prev_word)
                prev_word = cur_word
                prev_entity_group = cur_entity_group
    last_dct = {
        "word": prev_word,
        "entity_group": prev_entity_group
    }
    if prev_word not in final_words:
        final_entities.append(last_dct)
        final_words.add(prev_word)

    return final_entities



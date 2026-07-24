# Guardian NER & Article Classifier

NLP pipeline for Guardian articles: section classification, multi-label tag prediction and NER. Uses FastAPI.

## What it does

Given a Guardian article URL or raw text, the service returns:
- **section_id** - predicted section (e.g. `books`, `technology`, `environment`)
- **tags** - predicted content tags (e.g. `books/fiction`, `environment/climate-crisis`)
- **entities** - named entities extracted from the text (persons, organizations, locations, misc)

## Models

**Section & Tags — TF-IDF + Logistic Regression**

Text is lemmatized, vectorized with TF-IDF (unigrams + bigrams, top 5000 features), then classified.

| Task                    | Model                 | Eval Metric |
|-------------------------|-----------------------|---|
| Section ID (43 classes) | LogReg (L2, C=3.6)    | F1 macro = **0.829** |
| Tags (multi-label)      | OVR LogReg (L2, C=30) | F1 samples = **0.633** |

**NER — fine-tuned `dslim/bert-base-NER`**

Fine-tuned on 832 manually annotated Guardian sentences (train) + 200 (eval) in CoNLL format. Entities: `PER`, `ORG`, `LOC`, `MISC`. 
Best eval F1 = **0.836** (epoch 3)

## Project structure

```
src/
  api/
    main.py                        # FastAPI app
    log_reg_models_and_other_objects/  # serialized models and other objects
    ner_model/                     # fine-tuned NER model
  data_preparing/
    url_text_extraction.py         # fetch text from Guardian URL
    texts_preprocessing.py         # lemmatization + text prep
  ml/
    classifiers.py                 # section + tags prediction
    ner.py                         # NER inference
  notebooks/
    data_collection_and_filtering.ipynb  # notebook that was used to collect data and filter by section_id and etc
    ner.ipynb                            # notebook with NER model fine-tuning
    classifiers.py                       # notebook with LogReg training for section_id and tags prediction
```

## API endpoints

`POST /analyze/by_url` - fetch article from Guardian URL and analyze it.
```json
{"url": "https://www.theguardian.com/environment/2026/jul/20/some-article"}
```

`POST /analyze/by_text` - analyze raw title + body text directly.
```json
{"title": "Article title", "body_text": "Article body..."}
```

Example return:
```json
{
  "section_id": "environment",
  "tags": ["environment/climate-crisis", "science/science"],
  "entities": [{"word": "NASA", "entity_group": "ORG"}]
}
```

## Run with Docker

You can create Guardian API key at https://open-platform.theguardian.com/access/

```bash
docker build -t guardian-ner .
docker run -p 8000:8000 -e GUARDIAN_API_KEY="your_key" guardian-ner
```

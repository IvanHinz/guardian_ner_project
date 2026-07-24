import joblib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline

from data_preparing.url_text_extraction import get_article_text_by_url
from data_preparing.texts_preprocessing import prepare_main_text, preprocess_and_lemmatize_text
from ml.classifiers import predict_section, predict_tags, calculate_tf_idf
from ml.ner import collect_ner_entities


# To avoid problems with types
class ArticleRequest(BaseModel):
    url: str


class TextRequest(BaseModel):
    title: str
    body_text: str


class ArticleResponse(BaseModel):
    section_id: str | None
    tags: list[str]
    entities: list[dict]


models = {}
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before start - we load models and other objects (encoders, vectorizers)
    models_and_other_path_name = "log_reg_models_and_other_objects"
    ner_model_path = "ner_model"

    filenames = [
        "label_encoder",
        "log_reg_section_id",
        "tf_idf_vectorizer",
        "ovr_logreg_tags",
        "mlb",
    ]

    for filename in filenames:
        models[filename] = joblib.load(
            BASE_DIR / models_and_other_path_name / f"{filename}.joblib"
        )

    with open(BASE_DIR / models_and_other_path_name / "top_tags.json", "r", encoding="utf-8") as f:
        models["top_our_tags"] = json.load(f)

    models["ner_pipeline"] = pipeline(  # type: ignore
        "ner",
        model=BASE_DIR / ner_model_path,
        device="cpu",
        aggregation_strategy="simple",
    )

    yield

    # By shutdown
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/analyze/by_url", response_model=ArticleResponse)
async def analyze_article_by_url(request: ArticleRequest):
    request_url = request.url
    ans_dct = get_article_text_by_url(request_url)

    if ans_dct is None:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch the article. Check url or Guardian API availability!"
        )

    # Format title and body text in one text
    title, body_text = ans_dct["title"], ans_dct["body_text"]
    main_text = prepare_main_text(title, body_text)

    # NER Part
    main_ner_pipeline = models["ner_pipeline"]
    final_ner_entities = collect_ner_entities(main_ner_pipeline, main_text)

    return ArticleResponse(
        section_id=ans_dct["section_id"],
        tags=ans_dct["tags"],
        entities=final_ner_entities
    )


@app.post("/analyze/by_text", response_model=ArticleResponse)
async def analyze_by_text(request: TextRequest):
    main_text = prepare_main_text(request.title, request.body_text)

    # Classification part
    lemmatized_text = preprocess_and_lemmatize_text(main_text)

    tfidf_array = calculate_tf_idf(lemmatized_text, models["tf_idf_vectorizer"])

    # Predict section_id for text
    predicted_section_id = predict_section(
        tfidf_array,
        models["log_reg_section_id"],
        models["label_encoder"],
    )

    # Predict tags for text
    predicted_tags = predict_tags(
        tfidf_array,
        models["ovr_logreg_tags"],
        models["mlb"]
    )

    # NER Part
    main_ner_pipeline = models["ner_pipeline"]
    final_ner_entities = collect_ner_entities(main_ner_pipeline, main_text)

    return ArticleResponse(
        section_id=predicted_section_id,
        tags=predicted_tags,
        entities=final_ner_entities,
    )



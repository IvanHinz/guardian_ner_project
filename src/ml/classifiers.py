from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TAGS_THRESHOLD = 0.6


def calculate_tf_idf(
    text: str,
    tfidf: TfidfVectorizer,
):
    # Calculate TF-IDF
    tfidf_array = tfidf.transform([text])

    return tfidf_array


def predict_section(
        tfidf_array,
        logreg: LogisticRegression,
        section_encoder: LabelEncoder,
) -> str:
    # Logistic Regression prediction
    prediction_section_id = logreg.predict(tfidf_array)

    # Inverse transformation for LabelEncoder
    prediction_section_id_name = section_encoder.inverse_transform(prediction_section_id)[0]

    return prediction_section_id_name


def predict_tags(
    tfidf_array,
    logreg: LogisticRegression,
    mlb: MultiLabelBinarizer,
) -> list[str]:
    # Use logistic regression to predict where there are such tags
    prediction_tags_binary = (logreg.predict_proba(tfidf_array) > TAGS_THRESHOLD).astype(int)

    prediction_tags = mlb.inverse_transform(prediction_tags_binary)[0]

    prediction_tags = [el for el in prediction_tags]

    return prediction_tags


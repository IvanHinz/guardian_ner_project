import os
import requests
from dotenv import load_dotenv

load_dotenv()

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
GUARDIAN_ITEM_URL = "https://content.guardianapis.com/{article_id}"


def extract_article_id_from_url(guardian_url: str) -> str:
    url = guardian_url.split("?")[0]
    url_without_dom = url.replace("https://www.theguardian.com/", "")
    return url_without_dom


def get_article_text_by_url(guardian_url: str) -> dict | None:
    article_id = extract_article_id_from_url(guardian_url)

    response = requests.get(
        GUARDIAN_ITEM_URL.format(article_id=article_id),
        params={
            "api-key": GUARDIAN_API_KEY,
            "show-fields": "bodyText",
            "show-tags": "keyword",
        },
        timeout=30,
    )

    data = response.json()

    if "response" not in data:
        return None

    if data["response"]["status"] == "error":
        return None

    content = data["response"]["content"]

    ans_dct = {
        "title": content["webTitle"],
        "body_text": content.get("fields", {}).get("bodyText", ""),
        "section_id": content.get("sectionId", None),
        "section_name": content.get("sectionName", None),
        "tags": list(set(t["id"] for t in content.get("tags", []))),
        "tags_sections_ids": list(set(t.get("sectionId") for t in content.get("tags", []))),
        "tags_sections_names": list(set(t.get("sectionName") for t in content.get("tags", []))),
    }

    return ans_dct


# if __name__ == "__main__":
#     example_str = "https://www.theguardian.com/us-news/2026/jul/20/trump-canada-tariffs"
#
#     ex_dct = get_article_text_by_url(example_str)
#     print(ex_dct["tags"], ex_dct["tags_sections_ids"], ex_dct["tags_sections_names"])

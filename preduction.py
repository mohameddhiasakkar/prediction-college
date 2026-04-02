import json
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

_DATA_DIR = Path(__file__).resolve().parent
_DATA_FILE = _DATA_DIR / "contry.json"

with open(_DATA_FILE, encoding="utf-8") as f:
    universities = json.load(f)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/predict", methods=["OPTIONS"])
@app.route("/universities", methods=["OPTIONS"])
def preflight():
    return "", 204


def get_unis_by_country(country: str):
    needle = (country or "").lower()
    return [u for u in universities if needle in u.get("country", "").lower()]


def calculate_score(student: dict, uni: dict) -> int:
    score = 0
    student_country = (student.get("country") or "").lower()
    if student_country and student_country == (uni.get("country") or "").lower():
        score += 40

    skills = (student.get("skills") or "").lower()
    if "python" in skills:
        score += 30

    try:
        if float(student.get("moyenne") or 0) >= 12:
            score += 30
    except (TypeError, ValueError):
        pass

    return score


def recommend(student: dict):
    country = student.get("country") or ""
    filtered_unis = get_unis_by_country(country) if country else universities[:200]
    results = []
    for uni in filtered_unis[:50]:
        score = calculate_score(student, uni)
        results.append(
            {"name": uni["name"], "country": uni.get("country", ""), "score": score}
        )
    return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    result = recommend(data)
    return jsonify(result)


@app.route("/universities", methods=["GET"])
def list_universities():
    country = request.args.get("country", "")
    filtered = get_unis_by_country(country)
    return jsonify(filtered[:50])


if __name__ == "__main__":
    app.run(debug=True)

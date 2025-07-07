from flask import Flask, request, jsonify
from models.classifier import crisis_pipeline

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    report = data.get("report")
    if not report:
        return jsonify({"error": "Missing report"}), 400
    result = crisis_pipeline(report)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Blueprint, request, jsonify
from api.models.classifier import crisis_pipeline
from api.utils.supabase_client import insert_report_to_supabase

pipeline_bp = Blueprint("pipeline", __name__)

@pipeline_bp.route("/process", methods=["POST"])
def process_crisis():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    report_text = data["text"]
    result = crisis_pipeline(report_text)
    insert_report_to_supabase(result, report_text)
    return jsonify(result)

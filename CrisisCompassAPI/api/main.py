from flask import Flask
from api.routes.pipeline import pipeline_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(pipeline_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)

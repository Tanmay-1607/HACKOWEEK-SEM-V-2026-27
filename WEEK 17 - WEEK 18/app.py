from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

from pytorch_engine import (
    run_pytorch_basics_demo,
    run_cnn_demo,
    compare_sequence_models,
    compare_regularization_strategies,
    get_transfer_learning_summary,
    run_transformers_pipeline,
    execute_llm_task
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')
CORS(app)

# ─── Static Routing ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    if os.path.exists(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)
    return send_from_directory(PUBLIC_DIR, 'index.html')

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route('/api/pytorch/basics', methods=['GET'])
def api_pytorch_basics():
    res = run_pytorch_basics_demo()
    return jsonify({"success": True, "data": res})

@app.route('/api/cnn/demo', methods=['GET'])
def api_cnn_demo():
    res = run_cnn_demo()
    return jsonify({"success": True, "data": res})

@app.route('/api/sequence/compare', methods=['GET'])
def api_sequence_compare():
    res = compare_sequence_models()
    return jsonify({"success": True, "data": res})

@app.route('/api/regularization/compare', methods=['GET'])
def api_regularization_compare():
    res = compare_regularization_strategies()
    return jsonify({"success": True, "data": res})

@app.route('/api/transfer-learning/info', methods=['GET'])
def api_transfer_learning_info():
    res = get_transfer_learning_summary()
    return jsonify({"success": True, "data": res})

@app.route('/api/transformers/analyze', methods=['POST'])
def api_transformers_analyze():
    payload = request.get_json() or {}
    text = payload.get('text', '')
    res = run_transformers_pipeline(text)
    return jsonify({"success": True, "data": res})

@app.route('/api/llm/execute', methods=['POST'])
def api_llm_execute():
    payload = request.get_json() or {}
    provider = payload.get('provider', 'openai')
    model_name = payload.get('model', 'gpt-4o-mini')
    task = payload.get('task', 'generation')
    prompt = payload.get('prompt', 'Explain attention mechanism in Transformers')
    res = execute_llm_task(provider, model_name, task, prompt)
    return jsonify({"success": True, "data": res})

if __name__ == '__main__':
    print("=" * 65)
    print("[WEEK 17-18] PyTorch, CNNs, RNNs/LSTMs, Transfer Learning & LLMs Studio")
    print("[SERVER] Starting server at http://127.0.0.1:5008")
    print("=" * 65)
    app.run(debug=True, host='0.0.0.0', port=5008)

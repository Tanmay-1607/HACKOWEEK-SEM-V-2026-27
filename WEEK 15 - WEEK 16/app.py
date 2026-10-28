from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os

from neural_engine import (
    train_and_benchmark_deployment,
    predict_deployed_model,
    evaluate_logic_gates,
    get_activation_curves,
    compare_optimizers,
    train_custom_scratch_network
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

@app.route('/api/deployment/benchmark', methods=['GET'])
def api_deployment_benchmark():
    res = train_and_benchmark_deployment()
    return jsonify({"success": True, "data": res})

@app.route('/api/deployment/predict', methods=['POST'])
def api_deployment_predict():
    payload = request.get_json() or {}
    backend = payload.get('backend', 'joblib')
    input_data = payload.get('features', {})
    res = predict_deployed_model(input_data, backend=backend)
    return jsonify({"success": True, "data": res})

@app.route('/api/perceptron/gates', methods=['GET'])
def api_perceptron_gates():
    res = evaluate_logic_gates()
    return jsonify({"success": True, "data": res})

@app.route('/api/activations/curves', methods=['GET'])
def api_activations_curves():
    res = get_activation_curves()
    return jsonify({"success": True, "data": res})

@app.route('/api/optimizers/compare', methods=['GET'])
def api_optimizers_compare():
    res = compare_optimizers()
    return jsonify({"success": True, "data": res})

@app.route('/api/neuralnet/train', methods=['POST'])
def api_neuralnet_train():
    params = request.get_json() or {}
    res = train_custom_scratch_network(params)
    return jsonify({"success": True, "data": res})

if __name__ == '__main__':
    print("=" * 65)
    print("[WEEK 15-16] Neural Networks from Scratch & Model Deployment Studio")
    print("[SERVER] Starting server at http://127.0.0.1:5007")
    print("=" * 65)
    app.run(debug=True, host='0.0.0.0', port=5007)

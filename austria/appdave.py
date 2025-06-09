from flask import Flask, request, jsonify
import joblib
import time

app = Flask(__name__)

try:
    modello = joblib.load('insurance_model_pipeline.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    modello = None

@app.route('/infer', methods=['POST'])
def infer():
    start_time = time.time()
    data = request.get_json()

    try:
        valore = float(data["hyper_param_a"])
        if modello is None:
            raise ValueError("Model is not loaded")

        prediction = modello.predict([[valore]])
        latency = time.time() - start_time
        print(f"Inference time: {latency:.4f} seconds")

        return jsonify({"result": prediction[0], "latency_seconds": latency})

    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

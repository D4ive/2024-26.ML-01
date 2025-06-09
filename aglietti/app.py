from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    
    # Prendi il parametro come nell'immagine
    param = data.get('hyper_param_a', '')
    
    # Risposta semplice come nell'esempio
    return jsonify({
        "result": "A",
        "value": 123,
        "status": {
            "code": "OK",
            "model_latency": 0.12
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from agent import run_agent, initialize_vector_db
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    response, category = run_agent(user_input)
    return jsonify({"response": response, "category": category})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "online", "message": "Server is reachable!"})

@app.route('/reindex', methods=['POST'])
def reindex():
    success, message = initialize_vector_db(force_reindex=True)
    return jsonify({"success": success, "message": message})

if __name__ == '__main__':
    # Initialize DB on startup (not forcing reindex)
    # Only initialize in the main process, not the reloader
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        initialize_vector_db()
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import stock_engine

# Explicit template folder for robustness
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)
CORS(app)

# --- PAGE ROUTES ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/learn')
def learn(): return render_template('learn.html')

@app.route('/news')
def news(): return render_template('news.html')

@app.route('/login')
def login(): return render_template('login.html')

@app.route('/signup')
def signup(): return render_template('signup.html')

# --- API ROUTES ---
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        ticker = data.get('ticker', 'TCS')
        result = stock_engine.analyze_ticker(ticker)
        if result: return jsonify(result)
        return jsonify({"error": "Ticker not found. Try adding .NS"}), 404
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/market_status', methods=['GET'])
def market_status():
    try:
        result = stock_engine.analyze_ticker('^NSEI')
        if result:
            result['ticker'] = "NIFTY 50"
            return jsonify(result)
        return jsonify({"error": "Market data unavailable"}), 500
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_msg = request.json.get('message', '')
        response = stock_engine.get_chat_response(user_msg)
        return jsonify({"response": response})
    except: return jsonify({"response": "Connection error."})

@app.route('/api/news_analysis', methods=['GET'])
def get_news():
    try:
        return jsonify(stock_engine.get_ai_news())
    except: return jsonify([])

@app.route('/api/ticker_data', methods=['GET'])
def get_ticker_data():
    try:
        return jsonify(stock_engine.get_nifty_ticker_data())
    except: return jsonify([])

if __name__ == '__main__':
    print("\n--- AlphaFlow Server Started ---")
    print("✅ Server running on: http://127.0.0.1:5001")
    app.run(debug=True, port=5001)

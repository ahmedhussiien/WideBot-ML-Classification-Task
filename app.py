from flask import Flask, request, jsonify
from flask_cors import CORS
from serve import get_model_api 

app = Flask(__name__)
CORS(app) 
model_api = get_model_api()

# Default route
@app.route('/')
def index():
    return "Widebot bonary classifier API"

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

# API route
@app.route('/api', methods=['GET'])
def api():
    output_data = model_api(request)[0]
    response = jsonify(prediction = output_data)
    return response

if __name__ == '__main__':
    app.run(debug=True)
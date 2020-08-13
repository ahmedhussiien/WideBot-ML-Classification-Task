from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from serve import get_model_api 
import argparse

app = Flask(__name__)
CORS(app)
model_api = get_model_api()

### swagger specific ###
SWAGGER_URL = '/api/documentation'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "WideBot-Classifier-API"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###

# Default route
@app.route('/')
def home():
   return render_template('home.html')

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
    response = jsonify(predictedClass = output_data)
    return response


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(
        description="WideBot-Classifier-API")

    PARSER.add_argument('-d', '--debug', action='store_true',
                        help="Use flask debug/dev mode with file change reloading")
    ARGS = PARSER.parse_args()

    if ARGS.debug:
        print("Running in debug mode")
        app.run(debug=True)

    else:
        app.run(debug=False)
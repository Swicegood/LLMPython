# app.py 

from flask import Flask, request, jsonify
from llava_module import LLaVAModel
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
llava_model = LLaVAModel()

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    # Log the raw request
    logger.debug("Received request:")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Body: {request.get_data(as_text=True)}")

    try:
        data = request.json
        logger.debug(f"Parsed JSON data: {json.dumps(data, indent=2)}")
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return jsonify({"error": "Invalid JSON in request body"}), 400

    messages = data.get('messages', [])
    
    if not messages:
        logger.warning("No messages provided in the request")
        return jsonify({"error": "Invalid input. No messages provided."}), 400

    try:
        response_text, _ = llava_model.process_request(messages)

        if response_text is None:
            logger.error("LLaVA model returned None")
            return jsonify({"error": "Error processing request."}), 500

        response = {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llava-v1.6-mistral-7b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        logger.debug(f"Sending response: {json.dumps(response, indent=2)}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1234)
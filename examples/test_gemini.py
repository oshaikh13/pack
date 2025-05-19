# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types


def generate():

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "timestamp": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "caption": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
            },
        ),
    )

    client = genai.Client(api_key="GOOGLE_API_KEY")
    myfile = client.files.upload(file="path/to/sample.mp4")

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=[myfile, "PROPMT"],
        config=generate_content_config
    )

    print(response.text)

if __name__ == "__main__":
    generate()

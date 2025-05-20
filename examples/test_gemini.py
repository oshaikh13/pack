from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from google.genai import types as genai_types

def _sync_call() -> str:
    print("GENERATING CONFIG")
    generate_content_config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "timestamp": genai_types.Schema(type=genai_types.Type.STRING),
                "caption":   genai_types.Schema(type=genai_types.Type.STRING),
            },
        ),
    )

    video_path = "/Users/oshaikh/.cache/gum/screenshots/1747698197.60748.mp4"
    print("READING VIDEO INTO MEMORY")
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    print("INITIALIZING CLIENT")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print("GENERATING CONTENT")
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=genai_types.Content(
            parts=[
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        data=video_bytes,
                        mime_type="video/mp4"
                    )
                ),
                genai_types.Part(
                    text="Generate a caption for this video, where timestamp is in MM:SS format"
                ),
            ]
        ),
        config=generate_content_config,
    )

    print("RETURNING RESPONSE")
    print(resp.text)
    return resp.text

_sync_call()

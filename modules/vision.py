#이미지 분석과 관련된 모든 시각적 처리 기능을 담당하는 모듈입니다.
import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다. 경로: {image_path}")
        return None

def analyze_blurred_image(image_path: str) -> dict:
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return {"error": "이미지 인코딩에 실패했습니다."}

    prompt = """
    You are an expert AI assistant who analyzes the user's screen to understand their intent.
    Analyze the given image according to the instructions below and provide **only the final result in the specified JSON format**.

    ### Your Internal Thought Process (Chain of Thought):
    1.  **Observation:** First, observe all visual cues in the image in detail (e.g., application, window title, visible text, icons).
    2.  **Interpretation:** Based on the observed cues, interpret the user's core intent and what they are trying to accomplish.
    3.  **Synthesis:** Synthesize the observation and interpretation to create a specific, detailed user action scenario.
    
    **Important: Do not include your thought process in the final JSON output.**

    ### Final Output JSON Structure:
    - You must respond with only a valid JSON object.
    - The JSON object must have a single key: "current_action".
    - **The value for "current_action" must be a detailed paragraph written in Korean**, summarizing your internal analysis.
    
    ```json
    {
      "current_action": "내부 추론을 바탕으로 사용자의 현재 행동을 매우 구체적이고 상세하게 한글 문단으로 서술한 내용."
    }
    ```
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        response_content = response.choices[0].message.content
        if response_content.startswith("```json"):
            response_content = response_content.strip("```json\n").strip("```")
        
        return json.loads(response_content)

    except Exception as e:
        return {"error": f"API 호출 중 오류 발생: {e}"}

if __name__ == "__main__":
    test_image_path = "test_image_2.png"
    
    print(f"'{test_image_path}' 이미지 분석을 시작합니다...")
    analysis_result = analyze_blurred_image(test_image_path)
    
    print("\n--- 분석 결과 ---")
    print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
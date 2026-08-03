import os
import json
import urllib.request

# Detailed Fallback Knowledge Base for Offline/Cloned Environments
OFFLINE_ADVISORY_DATABASE = {
    "anthracnose": {
        "rehab_plan": [
            "Week 1: Prune infected foliage and incinerate diseased plant matter.",
            "Week 2: Apply Copper Hydroxide or Bordeaux mixture fungicide every 7 days.",
            "Week 3: Switch to drip irrigation to prevent standing water on leaves.",
            "Week 4: Apply organic neem-based protective spray and monitor new shoots."
        ],
        "organic_remedy": "Neem oil emulsion (5ml/L) + Potassium bicarbonate spray.",
        "prevention": "Ensure wide crop spacing for optimal canopy airflow and sunlight."
    },
    "bacterial blight": {
        "rehab_plan": [
            "Week 1: Isolate infected plants and stop overhead sprinkler watering.",
            "Week 2: Apply bactericide spray (Streptomycin sulphate + Copper oxychloride).",
            "Week 3: Soil drench with bio-fungicide Trichoderma viride.",
            "Week 4: Inspect new leaf flushes for bacterial oozing."
        ],
        "organic_remedy": "Pseudomonas fluorescens bio-agent soil treatment.",
        "prevention": "Use certified disease-free seed stock and crop rotation."
    },
    "brown spot": {
        "rehab_plan": [
            "Week 1: Apply Mancozeb 75% WP or Propiconazole 25% EC spray.",
            "Week 2: Apply balanced NPK fertilizer with additional potassium (K).",
            "Week 3: Clear surrounding wild grass weeds from field borders.",
            "Week 4: Repeat foliar spray if humidity remains >85%."
        ],
        "organic_remedy": "Fermented cow butter-milk spray or Panchagavya.",
        "prevention": "Avoid excessive nitrogen application; maintain soil potassium levels."
    },
    "fall armyworm": {
        "rehab_plan": [
            "Week 1: Deploy 10-12 pheromone lure traps per hectare.",
            "Week 2: Spray Bacillus thuringiensis (Bt) or Emamectin benzoate 5% SG.",
            "Week 3: Apply neem cake to crop soil whorls.",
            "Week 4: Monitor egg masses on underside of leaves."
        ],
        "organic_remedy": "Azadirachtin 10,000 ppm botanical insecticide.",
        "prevention": "Intercrop with flowering plants to attract natural parasitoid wasps."
    },
    "healthy": {
        "rehab_plan": [
            "Week 1-4: Plant is fully healthy! Maintain routine drip/soaker irrigation.",
            "Apply balanced organic compost and routine field scouting."
        ],
        "organic_remedy": "Routine Vermicompost & Panchagavya liquid fertilizer.",
        "prevention": "Standard crop rotation and regular monitoring."
    }
}

def query_agronomist_agent(disease_name, user_question=None):
    """
    Autonomous AI Agronomist Agent.
    Checks for GEMINI_API_KEY environment variable.
    If available, queries Gemini AI API. Otherwise, uses smart offline knowledge fallback.
    """
    disease_key = disease_name.lower().strip()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # If GEMINI_API_KEY is available, perform live GenAI Agent request
    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            prompt_text = f"You are an expert AI Agronomist Agent. The crop leaf diagnosis is '{disease_name}'. "
            if user_question:
                prompt_text += f"The farmer asks: '{user_question}'. "
            prompt_text += "Provide a clear 4-week rehabilitation plan, organic treatment recommendations, and preventative measures."

            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text_response = res_data['candidates'][0]['content']['parts'][0]['text']
                return {
                    "source": "Gemini 1.5 Flash AI Agent",
                    "response": text_response,
                    "is_live_ai": True
                }
        except Exception as e:
            print(f"[AI Agent] Live API call fallback: {e}")

    # Fallback / Offline Demo Mode (Guarantees zero crashes when cloned)
    default_info = OFFLINE_ADVISORY_DATABASE.get(disease_key, {
        "rehab_plan": [
            "Week 1: Remove severely affected leaves and isolate plant.",
            "Week 2: Apply recommended broad-spectrum bio-fungicide.",
            "Week 3: Improve soil drainage and ensure adequate sunlight exposure.",
            "Week 4: Monitor crop recovery and new growth."
        ],
        "organic_remedy": "Neem seed kernel extract (NSKE 5%) spray.",
        "prevention": "Consult local agricultural extension officer for regional spray schedules."
    })

    custom_answer = ""
    if user_question:
        custom_answer = f"**AI Answer to '{user_question}'**: Based on '{disease_name}', ensure proper soil drainage, avoid over-fertilization, and apply targeted copper/neem treatments."

    return {
        "source": "Autonomous AI Agronomist Agent (Demo Mode)",
        "rehab_plan": default_info["rehab_plan"],
        "organic_remedy": default_info["organic_remedy"],
        "prevention": default_info["prevention"],
        "custom_answer": custom_answer,
        "is_live_ai": False
    }

if __name__ == "__main__":
    res = query_agronomist_agent("anthracnose", "How often should I spray neem oil?")
    print("AI Agronomist Response:", json.dumps(res, indent=2))

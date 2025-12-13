
import json
import openai
import os

# Configuration
# Ideally this comes from env vars, but using the hardcoded key from writer.py for the hackathon
API_KEY = "sk-XhmNFaPbVqeIwl3RqrAvfQ"
BASE_URL = "https://imllm.intermesh.net/v1"
KB_PATH = "data/knowledge_base.json"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

class VoiceAgent:
    def __init__(self):
        self.kb = self._load_kb()
        
    def _load_kb(self):
        if not os.path.exists(KB_PATH):
            return []
        with open(KB_PATH, 'r') as f:
            return json.load(f)

    def find_relevant_sop(self, query):
        """
        Uses LLM to find the most relevant SOP from the Knowledge Base.
        """
        # Create a summary of available SOPs
        sop_summaries = []
        for item in self.kb:
             keywords = item.get('key_topics', [])
             # key_topics might be commented out or missing
             kw_str = f" (Keywords: {', '.join(keywords)})" if keywords else ""
             sop_summaries.append(f"ID {item['id']}: {item['concern_summary']}{kw_str}")
        sop_list_str = "\n".join(sop_summaries)
        
        system_prompt = f"""
        You are a classifier for an IndiaMART Customer Service Agent.
        Identify which SOP matches the user's query.
        
        Available SOPs:
        {sop_list_str}
        
        Return ONLY the ID of the matching SOP. If none match, return -1.
        """
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            # Extract number
            import re
            match = re.search(r'-?\d+', content)
            if match:
                sop_id = int(match.group())
                if sop_id != -1:
                    for sop in self.kb:
                        if sop['id'] == sop_id:
                            return sop
            return None
        except Exception as e:
            print(f"Error finding SOP: {e}")
            return None

    def get_response(self, query):
        sop = self.find_relevant_sop(query)
        
        if not sop:
            return "I am sorry, I am not trained to handle this specific query yet. Please explain in more detail or ask about something else."
        
        # Function-calling style generation for the response
        system_prompt = f"""
        You are an AI Voice Agent for IndiaMART.
        You are helping a seller.
        
        Current Issue: {sop['concern_summary']}
        
        SOP to Follow:
        {sop['resolution_sop']}
        
        Goal: Provide a helpful, natural response to the user's last message, strictly following the step-by-step SOP.
        If the user has just started, guide them through step 1.
        If they seem confused, explain the current step.
        """
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    # Test
    agent = VoiceAgent()
    print(agent.get_response("I want to sell wheat instead of rice."))

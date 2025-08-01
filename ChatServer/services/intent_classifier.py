"""
MCP Intent Classification Service
Uses Claude API to classify user intents and plan MCP tool executions
"""
import json
from typing import Dict, Any, List
from anthropic import Anthropic

class IntentClassifier:
    """
    Intent classifier that uses Claude to understand user requests and plan MCP tool executions
    """

    def __init__(self, anthropic_client: Anthropic):
        self.anthropic = anthropic_client
        self.supported_intents = [
            "search_papers",
            "save_paper",
            "list_papers",
            "get_paper_info",
            "help",
            "greeting",
            "unknown"
        ]

    async def classify_intent_and_plan(self, message: str, available_tools: List[Dict], user_id: str = None) -> Dict[str, Any]:
        """
        Classify user intent and create execution plan for MCP tools

        Args:
            message: User's input message
            available_tools: List of available MCP tools
            user_id: Optional user identifier

        Returns:
            Dictionary containing intent, plan, and initial response
        """

        # Create the classification prompt
        prompt = self._create_intent_classification_prompt(message, available_tools, user_id)

        try:
            # Call Claude API for intent classification and planning
            response = self.anthropic.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse the response
            result = self._parse_claude_response(response.content[0].text)

            print(f"🧠 Claude analysis complete: {result.get('intent', 'unknown')}")
            return result

        except Exception as e:
            print(f"❌ Intent classification failed: {str(e)}")
            return self._create_fallback_response(message)

    def _create_intent_classification_prompt(self, message: str, available_tools: List[Dict], user_id: str = None) -> str:
        """
        Create a comprehensive prompt for Claude to classify intent and plan actions
        """

        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])

        prompt = f"""You are an AI assistant that helps users with academic paper research. 

AVAILABLE MCP TOOLS:
{tools_description}

USER MESSAGE: "{message}"
USER ID: {user_id or "Anonymous"}

Please analyze the user's message and respond with a JSON object containing:

1. "intent": One of these categories:
   - "search_papers": User wants to find academic papers on a topic
   - "save_paper": User wants to save/bookmark a specific paper  
   - "list_papers": User wants to see their saved papers
   - "get_paper_info": User wants details about a specific paper
   - "help": User needs assistance or wants to know capabilities
   - "greeting": User is greeting or making small talk
   - "unknown": Intent unclear or not supported

2. "entities": Extract relevant information like:
   - search_terms: Topics, keywords, or subjects to search for
   - paper_id: Specific paper identifiers mentioned
   - author_names: Author names if mentioned
   - time_filters: Date ranges or publication years

3. "planned_actions": Array of MCP tool calls needed, each with:
   - "tool": Name of the MCP tool to call
   - "arguments": Parameters for the tool call
   - "reason": Why this tool call is needed

4. "response": Initial response to the user explaining what you'll do

5. "confidence": Confidence level (0.0 to 1.0) in the classification

EXAMPLES:

User: "search for machine learning papers"
Response:
{{
  "intent": "search_papers",
  "entities": {{"search_terms": "machine learning"}},
  "planned_actions": [
    {{
      "tool": "search_papers",
      "arguments": {{"topic": "machine learning", "max_results": 5}},
      "reason": "Search ArXiv for machine learning papers"
    }}
  ],
  "response": "I'll search for machine learning papers for you.",
  "confidence": 0.95
}}

User: "save paper 2401.12345"
Response:
{{
  "intent": "save_paper", 
  "entities": {{"paper_id": "2401.12345"}},
  "planned_actions": [
    {{
      "tool": "extract_info",
      "arguments": {{"paper_id": "2401.12345"}},
      "reason": "Get paper details before saving"
    }}
  ],
  "response": "I'll save paper 2401.12345 to your collection.",
  "confidence": 0.9
}}

User: "show my papers"
Response:
{{
  "intent": "list_papers",
  "entities": {{}},
  "planned_actions": [],
  "response": "Here are your saved papers: (This would typically fetch from database)",
  "confidence": 0.95
}}

Please respond with ONLY the JSON object, no additional text."""

        return prompt

    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude's JSON response

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed response dictionary
        """
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()

            # Find JSON block if wrapped in markdown
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            # Parse JSON
            result = json.loads(response_text)

            # Validate required fields
            if "intent" not in result:
                result["intent"] = "unknown"
            if "planned_actions" not in result:
                result["planned_actions"] = []
            if "response" not in result:
                result["response"] = "I understand your request."
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "entities" not in result:
                result["entities"] = {}

            return result

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Claude response as JSON: {e}")
            print(f"Raw response: {response_text}")
            return self._create_fallback_response_from_text(response_text)
        except Exception as e:
            print(f"❌ Error parsing Claude response: {e}")
            return self._create_fallback_response_from_text(response_text)

    def _create_fallback_response(self, message: str) -> Dict[str, Any]:
        """
        Create a fallback response when Claude API fails
        """
        message_lower = message.lower()

        # Simple keyword-based fallback
        if any(word in message_lower for word in ["search", "find", "look for", "papers about"]):
            return {
                "intent": "search_papers",
                "entities": {"search_terms": "general topic"},
                "planned_actions": [],
                "response": "I'll help you search for papers. However, I'm having trouble processing your request right now.",
                "confidence": 0.3
            }
        elif any(word in message_lower for word in ["save", "bookmark", "store"]):
            return {
                "intent": "save_paper",
                "entities": {},
                "planned_actions": [],
                "response": "I'll help you save a paper. Could you please specify the paper ID or title?",
                "confidence": 0.3
            }
        elif any(word in message_lower for word in ["list", "show", "my papers", "saved"]):
            return {
                "intent": "list_papers",
                "entities": {},
                "planned_actions": [],
                "response": "I'll show you your saved papers.",
                "confidence": 0.3
            }
        else:
            return {
                "intent": "unknown",
                "entities": {},
                "planned_actions": [],
                "response": "I'm having trouble understanding your request. Could you please rephrase it?",
                "confidence": 0.1
            }

    def _create_fallback_response_from_text(self, response_text: str) -> Dict[str, Any]:
        """
        Create fallback when JSON parsing fails but we have some response text
        """
        return {
            "intent": "unknown",
            "entities": {},
            "planned_actions": [],
            "response": f"I partially understood your request, but I'm having technical difficulties. Here's what I got: {response_text[:200]}...",
            "confidence": 0.2
        }

    async def generate_final_response(self, intent_result: Dict[str, Any], execution_results: List[Dict], original_message: str) -> str:
        """
        Generate final response based on intent and execution results

        Args:
            intent_result: Original intent classification result
            execution_results: Results from MCP tool executions
            original_message: User's original message

        Returns:
            Final formatted response
        """

        # Convert execution results to JSON-serializable format
        serializable_results = self._make_results_serializable(execution_results)

        prompt = f"""Based on the user's request and the results from tool executions, generate a helpful and natural response.

USER REQUEST: "{original_message}"
INTENT: {intent_result.get('intent', 'unknown')}

TOOL EXECUTION RESULTS:
{json.dumps(serializable_results, indent=2)}

Please provide a natural, helpful response that:
1. Acknowledges what the user asked for
2. Summarizes the results in a user-friendly way
3. Offers next steps or additional help if appropriate

Keep the response conversational and informative. If there were errors, explain them gently and suggest alternatives."""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"❌ Failed to generate final response: {e}")
            # Fallback to basic response formatting
            return self._format_results_simple(intent_result, execution_results)

    def _make_results_serializable(self, execution_results: List[Dict]) -> List[Dict]:
        """
        Convert MCP execution results to JSON-serializable format

        Args:
            execution_results: Raw execution results with TextContent objects

        Returns:
            JSON-serializable version of results
        """
        serializable_results = []

        for result in execution_results:
            serializable_result = {
                'action': result.get('action', {}),
                'success': result.get('success', False)
            }

            if result.get('success'):
                # Extract text content from MCP result
                raw_result = result.get('result', [])

                if isinstance(raw_result, list):
                    # Handle list of TextContent objects
                    text_results = []
                    for item in raw_result:
                        if hasattr(item, 'text'):
                            text_results.append(item.text)
                        elif isinstance(item, str):
                            text_results.append(item)
                        else:
                            text_results.append(str(item))
                    serializable_result['result'] = text_results

                elif hasattr(raw_result, 'text'):
                    # Single TextContent object
                    serializable_result['result'] = raw_result.text

                elif isinstance(raw_result, str):
                    # Already a string
                    serializable_result['result'] = raw_result

                else:
                    # Convert other types to string
                    serializable_result['result'] = str(raw_result)
            else:
                # Include error information
                serializable_result['error'] = result.get('error', 'Unknown error')

            serializable_results.append(serializable_result)

        return serializable_results

    def _format_results_simple(self, intent_result: Dict[str, Any], execution_results: List[Dict]) -> str:
        """
        Simple fallback formatting for execution results
        """
        if not execution_results:
            return intent_result.get('response', 'I completed your request.')

        successful_results = [r for r in execution_results if r.get('success')]
        failed_results = [r for r in execution_results if not r.get('success')]

        response = f"✅ Completed {len(successful_results)} operations successfully."

        if failed_results:
            response += f" ❌ {len(failed_results)} operations failed."

        # Add first successful result as example
        if successful_results:
            raw_result = successful_results[0].get('result', '')

            # Extract text content if it's TextContent objects
            if isinstance(raw_result, list):
                text_results = []
                for item in raw_result:
                    if hasattr(item, 'text'):
                        text_results.append(item.text)
                    else:
                        text_results.append(str(item))
                result_text = ', '.join(text_results)
            elif hasattr(raw_result, 'text'):
                result_text = raw_result.text
            else:
                result_text = str(raw_result)

            if len(result_text) > 0:
                response += f"\n\nResults: {result_text[:300]}..."

        return response
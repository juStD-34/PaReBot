"""
MCP Chat Server Main Entry Point
Integrates with MCP servers (ArXiv and PostgreSQL) for academic paper research
"""
import os
import asyncio
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from services.intent_classifier import IntentClassifier

# Load environment variables
load_dotenv()


class MCPChatServer:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.intent_classifier = IntentClassifier(self.anthropic)

        # MCP integration
        self.available_tools = []
        self.sessions = {}

    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        try:
            # Load server configuration
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})
            print(f"🔗 Connecting to {len(servers)} MCP servers...")

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

            print(f"✅ Connected to {len(self.sessions)} MCP servers")
            print(f"📋 Available tools: {[tool['name'] for tool in self.available_tools]}")

        except Exception as e:
            print(f"❌ Error loading server configuration: {e}")
            raise

    async def connect_to_server(self, server_name: str, server_config: dict):
        """Connect to a single MCP server"""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            for tool in response.tools:
                self.sessions[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })

            print(f"✅ Connected to {server_name} with tools: {[t.name for t in response.tools]}")

        except Exception as e:
            print(f"❌ Failed to connect to {server_name}: {e}")

    async def process_message(self, user_message: str, user_id: str = None) -> str:
        """
        Process incoming user message through intent classification and MCP tool execution

        Args:
            user_message: The user's input message
            user_id: Optional user identifier

        Returns:
            Generated response string
        """
        try:
            print(f"\n🔄 Processing message: '{user_message}'")

            # Step 1: Classify intent and determine actions
            intent_result = await self.intent_classifier.classify_intent_and_plan(
                user_message,
                available_tools=self.available_tools,
                user_id=user_id
            )

            print(f"🎯 Intent analysis: {intent_result['intent']}")
            print(f"📋 Planned actions: {len(intent_result.get('planned_actions', []))}")

            # Step 2: Execute planned actions using MCP tools
            if intent_result.get('planned_actions'):
                execution_results = await self.execute_planned_actions(intent_result['planned_actions'])
                print(f"Execution results: {(execution_results)}")
                # Step 3: Generate final response with execution results
                final_response = await self.intent_classifier.generate_final_response(
                    intent_result, execution_results, user_message
                )
            else:
                # No actions needed, use direct response
                final_response = intent_result.get('response', 'I understand, but I\'m not sure how to help with that.')

            return final_response

        except Exception as e:
            print(f"❌ Error processing message: {str(e)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."

    async def execute_planned_actions(self, planned_actions: list) -> list:
        """
        Execute the planned actions using appropriate MCP tools

        Args:
            planned_actions: List of actions to execute

        Returns:
            List of execution results
        """
        results = []

        for action in planned_actions:
            try:
                tool_name = action.get('tool')
                arguments = action.get('arguments', {})

                print(f"🔧 Executing tool: {tool_name} with args: {arguments}")

                # Get the appropriate session for this tool
                session = self.sessions.get(tool_name)
                if not session:
                    results.append({
                        'action': action,
                        'success': False,
                        'error': f"Tool '{tool_name}' not available"
                    })
                    continue

                # Execute the tool
                result = await session.call_tool(tool_name, arguments=arguments)

                results.append({
                    'action': action,
                    'success': True,
                    'result': result.content
                })

                print(f"✅ Tool execution successful")

            except Exception as e:
                print(f"❌ Tool execution failed: {str(e)}")
                results.append({
                    'action': action,
                    'success': False,
                    'error': str(e)
                })

        return results

    async def start_chat_loop(self):
        """Interactive chat loop"""
        print("\n🤖 MCP Academic Research Assistant Started!")
        print("I can help you search for papers, save them, and manage your research collection.")
        print("Available commands:")
        print("  • Search: 'search for machine learning papers'")
        print("  • Save: 'save this paper [paper_id]'")
        print("  • List: 'show my saved papers'")
        print("  • Help: 'help' or 'what can you do?'")
        print("Type 'quit' to exit.")
        print("-" * 70)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Happy researching!")
                    break

                if not user_input:
                    continue

                # Process the message
                response = await self.process_message(user_input, user_id="demo_user")
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    """Main entry point"""
    print("🚀 Initializing MCP Chat Server...")

    # Check environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        return

    chatserver = MCPChatServer()

    try:
        # Connect to MCP servers
        await chatserver.connect_to_servers()

        # Start interactive chat
        await chatserver.start_chat_loop()

    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
    finally:
        await chatserver.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
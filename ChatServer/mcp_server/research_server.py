"""
This is improvement version of research server - contain all 3 feature of MCP server:
    - tool: 2 tools existed in previous version
    - resource: 2 resources based on papers dir
    - prompt: 1 prompt
U still can run cmd 'npx @modelcontextprotocol/inspector uv run research_server.py' for server testing.
"""
import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP
import logging


logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see debug logs
    format="%(levelname)s [%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

PAPER_DIR = "../papers"

# Initialize FastMCP server
mcp = FastMCP("research")


@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """

    # Use arxiv to find the papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids

@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."

@mcp.resource("papers://folders")
def get_available_folder()->str:
    """
    :return: all available topic folders (child folders' name) in the papers directory
    """
    folders =[]
    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_dir_path = os.path.join(PAPER_DIR, topic_dir)
            if os.path.exists(topic_dir_path):
                papers_file = os.path.join(topic_dir_path, "papers_info.json")
                if os.path.exists(papers_file):
                    folders.append(topic_dir)
    content = "#Available Topics \n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
            content += f"\n Use @{folder} to access papers in that topic. \n"
    else :
        content += "No available topic folders."
    return content

@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
    :param topic: this is folder name of the topic
    :return: the papers related to the topic
    """
    topic_dir= topic.lower().replace(" ", "_")
    logger.debug(f"TOPIC: {topic_dir}")
    papers_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")
    logger.debug(f"Paper dir: {papers_file}")
    if not os.path.exists(papers_file):
        return f"No papers related to topic {topic}."
    try:
        with open(papers_file, "r") as f:
            papers_info = json.load(f)
        content = f"#Find {len(papers_info)} papers related to the topic " + topic + "\n"
        for paper_id, paper_info in papers_info.items():
            content += f"##{paper_info['title']}\n"
            content += f"- *Paper ID*: {paper_id}\n"
            content += f"- *Paper Authors*: {paper_info['authors']}\n"
            content += f"- *Paper Published* : {paper_info['published']}\n"
            content += f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"
        return content
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.exception(f"Error reading {papers_file}: {str(e)}")
        return f"There are error reading topic {topic}."


@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
    1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers})
    2. For each paper found, extract and organize the following information:
       - Paper title
       - Authors
       - Publication date
       - Brief summary of the key findings
       - Main contributions or innovations
       - Methodologies used
       - Relevance to the topic '{topic}'

    3. Provide a comprehensive summary that includes:
       - Overview of the current state of research in '{topic}'
       - Common themes and trends across the papers
       - Key research gaps or areas for future investigation
       - Most impactful or influential papers in this area

    4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.

    Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""
if __name__ == "__main__":
    mcp.run(transport='stdio')


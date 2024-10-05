import importlib.util
import sys
import json
import re
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import Union, List
from enum import Enum
import mechanicalsoup
from urllib.parse import urlparse
import re
import argparse
import random

# Global Variables
OPENAI_API_KEY = json.load(open('secrets.json', 'r'))['OPENAI_KEY']
BAD_KEYWORDS = ['support.google', 'accounts.google', 'maps.google']
PINTEREST_DOWNLOADER_PATH = './pinterest-downloader/pinterest-downloader.py'
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_MAX_CALLS = 5
OUTPUT_FILE = "products.jsonl"

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Parse Pinterest board URL to extract board name
def parse_board_name(url: str) -> str:
    parsed_url = urlparse(url)
    path_match = re.search(r"\.com/([^/]+)/([^/]+)/", parsed_url.geturl())
    if path_match:
        return f"{path_match.group(1)}/{path_match.group(2)}"
    else:
        raise ValueError("Invalid Pinterest board URL format")

# Function to get image URLs from Pinterest board
def get_image_urls(board_name: str) -> List[str]:
    kwargs = {'arg_path': board_name, 'arg_dir': 'images', 'arg_thread_max': 0,
              'arg_cut': -1, 'arg_board_timestamp': False, 'arg_log_timestamp': False,
              'arg_force': False, 'arg_exclude_section': False, 'arg_rescrape': False,
              'arg_img_only': False, 'arg_v_only': False, 'arg_update_all': False,
              'arg_https_proxy': None, 'arg_http_proxy': None, 'arg_cookies': None}

    # Load Pinterest downloader module
    spec = importlib.util.spec_from_file_location("pinterest_downloader_module", PINTEREST_DOWNLOADER_PATH)
    pinterest_downloader_module = importlib.util.module_from_spec(spec)
    sys.modules["pinterest_downloader_module"] = pinterest_downloader_module
    spec.loader.exec_module(pinterest_downloader_module)

    # Get image URLs from the Pinterest board
    return pinterest_downloader_module.run_library_main(**kwargs)

# Define Step and SearchQueryGeneration models
class Step(BaseModel):
    explanation: str
    output: str

class SearchQueryGeneration(BaseModel):
    steps: list[Step]
    final_answer: list[str]

# Generate search queries from moodboard images
async def generate_search_queries(image_urls: List[str], sample_size: int = DEFAULT_SAMPLE_SIZE, max_calls: int = DEFAULT_MAX_CALLS):
    calls = min(max_calls, (len(image_urls) + sample_size - 1) // sample_size)
    tasks = []

    for i in range(calls):
        sampled_urls = random.sample(image_urls, min(sample_size, len(image_urls)))
        tasks.append(
            client.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a shopping assistant. Based on the mood board images provided, "
                            "generate a list of search queries to help the user find similar products. "
                            "Explain your reasoning step by step."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Based on these images, what search queries should I use to find similar products?"},
                        ] + [
                            {"type": "image_url", "image_url": {"url": i}} for i in sampled_urls
                        ],
                    },
                ],
                response_format=SearchQueryGeneration,
            )
        )

    results = await asyncio.gather(*tasks)
    search_queries = [result.choices[0].message.parsed.final_answer for result in results]
    return [query for sublist in search_queries for query in sublist]

# Perform Google search for products
async def google_search(queries: List[str]):
    target_urls = []
    tasks = []
    for query in queries:
        browser = mechanicalsoup.StatefulBrowser()
        browser.open("https://www.google.com/")
        browser.select_form('form[action="/search"]')
        browser["q"] = f'{query} "price" "add to cart"'
        browser.submit_selected(btnName="btnG")
        for link in browser.links():
            target = link.attrs['href']
            if (target.startswith('/url?') and not target.startswith("/url?q=http://webcache.googleusercontent.com")):
                target = re.sub(r"^/url\?q=([^&]*)&.*", r"\1", target)
                if target.startswith('https') and not any(i in target for i in BAD_KEYWORDS):
                    target_urls.append(target)
    return list(set(target_urls))

# Asynchronous function to fetch HTML content from URLs
async def fetch_html(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text(), url
            else:
                return None, url
    except Exception as e:
        print(f"Failed to fetch URL {url}: {e}")
        return None, url

# Clean HTML content
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    unuseful_classes = ['ad-banner', 'popup', 'newsletter-signup']
    unuseful_ids = ['sidebar', 'comments', 'related-articles']
    for cls in unuseful_classes:
        for tag in soup.find_all(class_=cls):
            tag.decompose()
    for id_ in unuseful_ids:
        for tag in soup.find_all(id_=id_):
            tag.decompose()
    return soup.prettify()

# Define ProductExtraction and HtmlExtraction models
class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    FAIL = "fail"

class ProductExtraction(BaseModel):
    product_name: str
    product_description: str
    material: str = None
    dimensions_size: str = None
    price: float
    currency: str
    availability: str
    image_url: str
    category: str
    color: str = None
    product_url: str

class HtmlExtraction(BaseModel):
    status: ExtractionStatus
    product: Union[ProductExtraction, None] = None
    fail_reason: str = None

# Extract product details from HTML using OpenAI
async def extract_product_details(html_content: str, url: str):
    system_prompt = (
        "You are an expert in extracting structured e-commerce data. "
        "You will be provided with HTML text from an e-commerce website. "
        "Your task is to parse the HTML and extract details such as product name, price, image, and other specifications "
        "into a predefined structure."
    )
    completion = await client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": html_content}
        ],
        response_format=HtmlExtraction,
    )
    extraction_result = completion.choices[0].message.parsed
    if extraction_result.status == ExtractionStatus.SUCCESS:
        extraction_result.product.product_url = url
    return extraction_result

# Main function to execute the workflow
async def main():
    parser = argparse.ArgumentParser(description="Pinterest Moodboard Product Recommender")
    parser.add_argument('--pinterest_url', type=str, help='Pinterest moodboard URL')
    parser.add_argument('--sample_size', type=int, default=DEFAULT_SAMPLE_SIZE, help='Number of images to sample per query generation call')
    parser.add_argument('--max_calls', type=int, default=DEFAULT_MAX_CALLS, help='Maximum number of query generation calls')
    args = parser.parse_args()

    if not args.pinterest_url:
        args.pinterest_url = input("Please enter the Pinterest moodboard URL: ")

    pinterest_url = args.pinterest_url
    print(f"Parsing board name from URL: {pinterest_url}")
    board_name = parse_board_name(pinterest_url)
    print(f"Board name extracted: {board_name}")

    print("Fetching image URLs from the Pinterest board...")
    image_urls = get_image_urls(board_name)
    print(f"Fetched {len(image_urls)} image URLs.")

    print("Generating search queries from moodboard images...")
    queries = await generate_search_queries(image_urls, sample_size=args.sample_size, max_calls=args.max_calls)
    print(f"Generated {len(queries)} search queries.")

    print("Performing Google search for products...")
    target_urls = await google_search(queries)
    print(f"Found {len(target_urls)} unique product URLs.")

    async with aiohttp.ClientSession() as session:
        print("Fetching HTML content from product URLs...")
        html_tasks = [fetch_html(session, url) for url in target_urls]
        html_results = await asyncio.gather(*html_tasks)
        
        clean_html_results = [(clean_html(html), url) for html, url in html_results if html is not None]
        
        print("Extracting product details from HTML content...")
        extraction_tasks = [extract_product_details(html, url) for html, url in clean_html_results]
        extraction_results = await asyncio.gather(*extraction_tasks)
        
        products = [result.product for result in extraction_results if result.status == ExtractionStatus.SUCCESS]
        print(f"Extracted {len(products)} products.")

        # Save products to a JSONL file
        with open(OUTPUT_FILE, 'w') as f:
            for product in products:
                f.write(json.dumps(product.dict()) + "\n")
        print(f"Saved product details to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
import logging
import os
import pathlib
import re
import urllib.request
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

log = logging.getLogger(__name__)

# Regex pattern to match a URL
HTTP_URL_PATTERN = r"^http[s]*://.+"


# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add
        # the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

            # Function to get the hyperlinks from a URL


def get_hyperlinks(url):
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:
            # If the response is not HTML, return an empty list
            if not response.info().get("Content-Type").startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode("utf-8")
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks


def get_domain_hyperlinks(local_domain, url):
    """
    get the hyperlinks from a URL that are within the same domain
    """
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def crawl(url):
    """
    Crawls a given URL and returns a list of all the URLs
    found on the same domain.

    Parameters:
    url (str): The URL to crawl.

    Returns:
    list: list of URLs found on the same domain as the given URL.
    """
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Store URLs that have already been seen in a set no duplicates
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/" + local_domain + "/"):
        os.mkdir("text/" + local_domain + "/")

    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url = queue.pop()
        log.info(url)  # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        file_name = pathlib.Path(
            "text/" + local_domain + "/" + url[8:].replace("/", "_") + ".txt"
        )
        if not file_name.exists():
            with open(
                file_name,
                "w",
                encoding="UTF-8",
            ) as f:
                # Get the text from the URL using BeautifulSoup
                soup = BeautifulSoup(requests.get(url).text, "html.parser")

                # Get the text but remove the tags
                text = soup.get_text()

                # If the crawler gets to a page that requires
                # JavaScript, it will stop the crawl
                if "You need to enable JavaScript to run this app." in text:
                    print(
                        "Unable to parse page " + url + ", JavaScript required"
                    )

                # Otherwise, write the text to the file in the text directory
                f.write(text)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)


def main():
    full_url = "https://openai.com/"  # domain to be crawled with https or http
    crawl(full_url)
    text_dir = pathlib.Path(SCRIPT_DIR / "text")
    text_files = list((text_dir / "openai.com").glob("*"))
    log.info(f"There are {len(text_files)} text files downloaded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

from bs4 import BeautifulSoup


def is_valid_html(html_str: str) -> bool:
    return bool(BeautifulSoup(html_str, "html.parser").find())

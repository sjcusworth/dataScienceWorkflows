import sys
import re


def remove_commas_within_quotes(text):
    # Use regular expression to find all occurrences of text within double quotes
    matches = re.findall(r'(\".*?\")', text)

    # Replace commas within each match
    for match in matches:
        new_match = match.replace(',', '')
        new_match = new_match.replace('"', '')
        text = text.replace(match, new_match)

    return text

# Example usage
for x in sys.stdin:
    sys.stdout.write(remove_commas_within_quotes(x))

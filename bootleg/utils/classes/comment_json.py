"""
An example of how to remove comments and trailing commas from JSON before
parsing.  You only need the two functions below, `remove_comments()` and
`remove_trailing_commas()` to accomplish this.  This script serves as an
example of how to use them but feel free to just copy & paste them into your
own code/projects.  Usage::
    json_cleaner.py some_file.json
Alternatively, you can pipe JSON into this script and it'll clean it up::
    cat some_file.json | json_cleaner.py
Why would you do this?  So you can have human-generated .json files
(say, for configuration) that include comments and, really, who wants to deal
with catching all those trailing commas that might be present?  Here's an
example of a file that will be successfully cleaned up and JSON-parseable:
.. code-block:: javascript
    {
        // A comment!  You normally can't put these in JSON
        "testing": {
            "foo": "bar", // <-- A trailing comma!  No worries.
        }, // <-- Another one!
        /*
        This style of comments will also be safely removed before parsing
        */
    }
FYI:  This script will also pretty-print the JSON after it's cleaned up (if
using it from the command line) with an indentation level of 4 (that is, four
spaces).
"""

__version__ = '1.0.0'
__version_info__ = (1, 0, 0)
__license__ = "Unlicense"
__author__ = 'Dan McDougall <daniel.mcdougall@liftoffsoftware.com>'

import re

def remove_comments(json_like):
    """
    Removes C-style comments from *json_like* and returns the result.  Example::
        >>> test_json = '''\
        {
            "foo": "bar", // This is a single-line comment
            "baz": "blah" /* Multi-line
            Comment */
        }'''
        >>> remove_comments('{"foo":"bar","baz":"blah",}')
        '{\n    "foo":"bar",\n    "baz":"blah"\n}'
    """
    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    def replacer(match):
        s = match.group(0)
        if s[0] == '/': return ""
        return s
    return comments_re.sub(replacer, json_like)

def remove_trailing_commas(json_like):
    """
    Removes trailing commas from *json_like* and returns the result.  Example::
        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_like)
    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)
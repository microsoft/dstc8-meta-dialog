import re

# Common ===============================================================================================================
WHITESPACE_RGX = re.compile(r'\s+')

TIME_RGX = re.compile(
    r'(\b((\d{1,2}(\.\d\d)?\s*(am|pm|\s+o\s*\'?\s*clock))|(\d{1,2}[\.:]\d\d)))', re.I)
TIME_TOKEN = r'<time>'

WEEKDAY_RGX = re.compile(
    r'(\b(mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)', re.I)
WEEKDAY_TOKEN = r'<weekday>'

# months excluding may (too many false positives in goal oriented dialogue)
MONTH_RGX = re.compile(
    r'(\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|january|february|march|'
    r'april|june|july|august|september|october|november|december)\b)', re.I)
MONTH_TOKEN = r'<month>'

ORDINAL_RGX = re.compile(r'(\b(\d+(th|rd|nd)|\d+\.))', re.I)
ORDINAL_TOKEN = r' <ordinal> '

NUMBER_RGX = re.compile(r'(\b\d+\b)', re.I)
NUMBER_TOKEN = r' <num> '

EMAIL_RGX = re.compile(r'(\b\w[^\s]+@\w[\w\.-_]*\.\w+\b)')
EMAIL_TOKEN = r'<email>'

URL_RGX = re.compile(
    r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
URL_TOKEN = r'<url>'

ALPHANUM_RGX = re.compile(r'[a-zA-Z0-9]')

# Reddit ===============================================================================================================

SELF_BREAK_TOKEN = r'<selfbr>'
SELF_BREAK_RGX = re.compile(SELF_BREAK_TOKEN)

GET_SUBMISSION_TITLE_RGX = re.compile(
    r'^.*?(?=%s)' % SELF_BREAK_TOKEN, re.DOTALL)
GET_SUBMISSION_SELF_TEXT_RGX = re.compile(
    r'(?<=%s).*$' % SELF_BREAK_TOKEN, re.DOTALL)

LINE_BREAK_RGX = re.compile(r'((\n|\r\n)+)')
LINE_BREAK_TOKEN = r'<br>'

BOT_BODY_RGX = re.compile(
    r'^i am a bot|^i\'m a bot|^bleep.*?bloop|^beep.*?boop|i am a bot[^a-zA-Z]*$|^i\'m a bot[^a-zA-Z]*$|bleep.*?bloop[^a-zA-Z]*$|'
    r'beep.*?boop[^a-zA-Z]*$', re.I)
BOT_BUTTON_RGX = re.compile(r'\^\|\s*\^\[')
BOT_AUTHOR_PATTERNS = [
    r'^imgur',
    r'^linkfixer',
    r'bots?[^a-zA-Z]*$',
    r'tips?$',
    r'quotes$',
    r'transcriber$',
    r'watch$',
    r'breaker$',
    r'fixer$',
]
BOT_AUTHOR_RGX = re.compile('|'.join(BOT_AUTHOR_PATTERNS), re.I)

QUOTE_REPLACE_RGX = re.compile(r'&gt;(.*?)\n\n', re.DOTALL)
# In case we ever want to keep the quoted content
QUOTE_REPLACE_STRING = r'<quote> \g<1> </quote>'

USER_RGX = re.compile(r'(\@[\w\-]+)|(\/u\/\w+)', re.I)
USER_TOKEN = r'<mention>'

SUBREDDIT_RGX = re.compile(r'\/r\/\w+', re.I)
SUBREDDIT_TOKEN = r'<loc>'

TAG_RGX = re.compile(r'\[.*?\]')
TAG_TOKEN = r'<tag>'

# Markdown tables: https://www.markdownguide.org/extended-syntax/#tables
DETECT_MARKDOWN_TABLE_RGX = re.compile(r'(\|\s*:?--*:?\s*\|)|(\+----*)')

# HTML entities
HTML_NBSP_RGX = re.compile(r'\&nbsp\;')
HTML_GT_RGX = re.compile(r'\&gt\;')
HTML_LT_RGX = re.compile(r'\&lt\;')
HTML_AMP_RGX = re.compile(r'\&amp\;')
HTML_SINGLE_QUOTE_RGX = re.compile(r'\&apos\;')
HTML_DOUBLE_QUOTE_RGX = re.compile(r'\&quot\;')
HTML_ENTITY_RGX = re.compile(r'\&\w+\;')

# Special reddit markdown
# - word boundaries aren't enough inside encased text e.g. ~~[text]~~ is valid
# - use non greedy qualifiers since reddit editor+renderer should escape special chars e.g. **hello \*\* world**
# - can't do superscripts inside superscripts

# not needed if we render to HTML first. may not be worth it.
EMPH_RGX = re.compile(r'(?<!\\)\*+(.*?)(?<!\\)\*+')
EMPH_REPLACEMENT = r'\g<1>'

# may not be worth it
SUPERSRIPT1_RGX = re.compile(r'\b\^\b')
SUPERSRIPT1_REPLACEMENT = ' '
SUPERSRIPT2_RGX = re.compile(r'\^\(([^\^]+)\)')
SUPERSRIPT2_REPLACEMENT = r'\g<1>'

LINK_RGX = re.compile(r'(?<!\\)\[(.*?)(?<!\\)\]\(.*?\)')
LINK_REPLACEMENT = r'\g<1>'

STRIKETHROUGH_RGX = re.compile(r'~~(?=\S)(.*?)(?=\S)~~')
STRIKETHROUGH_REPLACEMENT = ''

SPOILER_RGX = re.compile(r'\>\!(?=\S)(.*?)(?=\S)\!\<')
SPOILER_REPLACEMENT = r'\g<1>'

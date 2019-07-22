import re

import mldc.preprocessing.norm_constants as N


class NormalizationError(RuntimeError):
  pass


class TextNormalizer(object):

  def __init__(self, token_limit=-1, char_limit=140, max_token_length=30):

    self.token_limit = token_limit
    self.char_limit = char_limit
    self.max_token_length = max_token_length
    self.char_limit_rgx = None

    if self.char_limit > 0:
      self.char_limit_rgx = re.compile(r'.{0,%d}\w\b' % (self.char_limit - 1), re.DOTALL)

  # `keep_end` used to mean start counting from the end rather than the beginning, ie truncate from the start
  def _limit_chars_on_word_boundary(self, text):
    if self.char_limit_rgx is None or self.char_limit <= 0:
      raise RuntimeError(
        'No char limit specified so can''t limit chars!')

    if len(text) <= self.char_limit:
      return text

    match = self.char_limit_rgx.match(text)

    if match is None:
      return text[:self.char_limit]

    return match.group()

  def _apply_limit_to_content(self, content):
    if self.token_limit > 0:
      content = ' '.join(content.split()[self.turn_limit:])
    elif self.char_limit > 0:
      content = self._limit_chars_on_word_boundary(content)

    if self.max_token_length > 0:
      content = ' '.join([t[:self.max_token_length] for t in content.split()])

    return content

  def __call__(self, raw_content):

    content = self._apply_limit_to_content(raw_content)

    if not content:
      raise NormalizationError(f"Normalized `{raw_content}` has zero length!`")

    return content


class MetalwozTextNormalizer(TextNormalizer):

  def __call__(self, raw_content):

    content = super().__call__(raw_content)

    content = N.TIME_RGX.sub(N.TIME_TOKEN, content)
    content = N.WEEKDAY_RGX.sub(N.WEEKDAY_TOKEN, content)
    content = N.MONTH_RGX.sub(N.MONTH_TOKEN, content)
    content = N.ORDINAL_RGX.sub(N.ORDINAL_TOKEN, content)
    content = N.EMAIL_RGX.sub(N.EMAIL_TOKEN, content)
    content = N.NUMBER_RGX.sub(N.NUMBER_TOKEN, content)
    content = N.URL_RGX.sub(N.URL_TOKEN, content)
    content = N.WHITESPACE_RGX.sub(' ', content)
    content = content.strip()

    if not content:
      raise NormalizationError(f"Normalized `{raw_content}` has zero length!`")

    return content


class RedditTextNormalizer(TextNormalizer):

  def __call__(self, raw_content):

    # Only limit the self text
    selftext_match = N.GET_SUBMISSION_SELF_TEXT_RGX.search(raw_content)
    if selftext_match:
      title = N.GET_SUBMISSION_TITLE_RGX.search(raw_content).group().strip()
      selftext = selftext_match.group().strip()
      content = selftext
    else:
      title = ''
      content = raw_content

    content = super().__call__(content)

    # If the turn was a submission, and has self text after limiting, add back a separator
    if title and content:
      sep = ' %s ' % N.LINE_BREAK_TOKEN
      content = title + sep + content
    elif title:
      content = title

    # Deal with reddit-specific stuff first
    # Get rid of quoted passages
    content = N.QUOTE_REPLACE_RGX.sub('', content)
    # Get rid of links
    content = N.LINK_RGX.sub(N.LINK_REPLACEMENT, content)
    content = N.TAG_RGX.sub('', content)
    content = N.USER_RGX.sub(N.USER_TOKEN, content)
    content = N.SUBREDDIT_RGX.sub(N.SUBREDDIT_TOKEN, content)
    content = N.EMPH_RGX.sub(N.EMPH_REPLACEMENT, content)
    content = N.SUPERSRIPT1_RGX.sub(N.SUPERSRIPT1_REPLACEMENT, content)
    content = N.SUPERSRIPT2_RGX.sub(N.SUPERSRIPT2_REPLACEMENT, content)
    content = N.STRIKETHROUGH_RGX.sub(N.STRIKETHROUGH_REPLACEMENT, content)
    content = N.SPOILER_RGX.sub(N.SPOILER_REPLACEMENT, content)

    # Replace other numerical data, urls, emails, etc
    content = N.TIME_RGX.sub(N.TIME_TOKEN, content)
    content = N.WEEKDAY_RGX.sub(N.WEEKDAY_TOKEN, content)
    content = N.MONTH_RGX.sub(N.MONTH_TOKEN, content)
    content = N.ORDINAL_RGX.sub(N.ORDINAL_TOKEN, content)
    content = N.EMAIL_RGX.sub(N.EMAIL_TOKEN, content)
    content = N.NUMBER_RGX.sub(N.NUMBER_TOKEN, content)
    content = N.URL_RGX.sub(N.URL_TOKEN, content)

    # Replace HTML entities
    content = N.HTML_NBSP_RGX.sub(' ', content)
    content = N.HTML_GT_RGX.sub('>', content)
    content = N.HTML_LT_RGX.sub('<', content)
    content = N.HTML_AMP_RGX.sub('&', content)
    content = N.HTML_SINGLE_QUOTE_RGX.sub('\'', content)
    content = N.HTML_DOUBLE_QUOTE_RGX.sub('"', content)
    content = N.HTML_ENTITY_RGX.sub('', content)

    # Condense whitespace. Must be done after quotes & markdown
    content = N.LINE_BREAK_RGX.sub(' %s ' % N.LINE_BREAK_TOKEN, content)
    content = N.WHITESPACE_RGX.sub(' ', content)
    content = content.strip()

    if not content:
      raise NormalizationError(f"Normalized `{raw_content}` has zero length!`")

    return content

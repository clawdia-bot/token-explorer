"""Helpers for consistent dark-mode HTML output."""

import re


DARK_MODE_HEAD = """
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#151515">
<style>
  :root { color-scheme: dark; }
  html { background: #151515; }
  body {
    background: #151515 !important;
    color: #e8e8e8;
  }
  a { color: #8ab4ff; }
  * { forced-color-adjust: none; }
</style>
""".strip()


def inject_dark_mode(html: str) -> str:
    """Add native dark-mode hints and page shell styling to an HTML document."""
    if 'name="color-scheme"' not in html:
        html = html.replace('<head>', '<head>\n' + DARK_MODE_HEAD + '\n', 1)

    if '<body>' in html:
        html = html.replace(
            '<body>',
            '<body style="background:#151515;color:#e8e8e8;">',
            1,
        )
    else:
        html = re.sub(
            r'<body([^>]*)>',
            r'<body\1 style="background:#151515;color:#e8e8e8;">',
            html,
            count=1,
        )

    return html

"""
Console helper — works with or without the 'rich' library installed.
Falls back to standard print() if rich is unavailable.
"""

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt

    _console = RichConsole()

    def print_msg(msg, style=None):
        try:
            _console.print(msg)
        except UnicodeEncodeError:
            try:
                # Replace unsupported characters with ascii equivalents
                import re as _re
                # Remove common rich tags for raw fallback printing
                clean_msg = _re.sub(r'\[/?[^\]]*\]', '', str(msg))
                safe_msg = clean_msg.encode('ascii', 'replace').decode('ascii')
                # Replace common symbols
                safe_msg = safe_msg.replace('?', ' ')  # replace the question mark if it was a unicode char
                print(safe_msg)
            except Exception:
                pass

    def print_panel(content, title=""):
        try:
            _console.print(Panel(content, title=title))
        except UnicodeEncodeError:
            try:
                import re as _re
                clean_content = _re.sub(r'\[/?[^\]]*\]', '', str(content))
                safe_content = clean_content.encode('ascii', 'replace').decode('ascii')
                print(f"\n--- {title} ---")
                print(safe_content)
                print("-" * (len(title) + 8) + "\n")
            except Exception:
                pass

    def make_table(title=""):
        return Table(title=title)

    def prompt_user(label):
        return Prompt.ask(label)

    RICH_AVAILABLE = True

except ImportError:
    import re

    def _strip_markup(text):
        """Remove rich markup tags like [bold], [cyan], etc."""
        return re.sub(r'\[/?[^\]]*\]', '', text)

    def print_msg(msg, style=None):
        print(_strip_markup(str(msg)))

    def print_panel(content, title=""):
        clean = _strip_markup(str(content))
        border = "═" * 50
        print(f"\n{border}")
        if title:
            print(f"  {title}")
            print(f"{'─' * 50}")
        print(clean)
        print(f"{border}\n")

    class _FakeTable:
        def __init__(self, title=""):
            self.title = title
            self.rows = []
        def add_column(self, name, **kwargs):
            pass
        def add_row(self, *args):
            self.rows.append(args)

    def make_table(title=""):
        return _FakeTable(title=title)

    def prompt_user(label):
        clean = _strip_markup(str(label))
        return input(f"{clean}: ")

    RICH_AVAILABLE = False


try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, desc=""):
        return _tqdm(iterable, desc=desc)
except ImportError:
    def tqdm(iterable, desc=""):
        if desc:
            print(f"{desc}...")
        return iterable

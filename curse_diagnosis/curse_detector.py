from better_profanity import profanity


class CurseDetector:
    def __init__(self):
        self.profanity = profanity

    def detect_curses(self, text):
        words = text.split()
        detected_curses = [word for word in words if self.profanity.contains_profanity(word)]
        return ', '.join(detected_curses) if detected_curses else None

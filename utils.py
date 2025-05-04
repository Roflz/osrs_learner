import threading
import difflib


def snap_text(text, choices, cutoff=0.6):
    """
    Return the closest match from choices to the input text, or the original text if no match.
    Useful for snapping user-entered labels to valid SKILLS.
    """
    matches = difflib.get_close_matches(text, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else text


def run_in_thread(fn, *args, daemon=True, **kwargs):
    """
    Run the given function in a background thread.
    Returns the Thread object.
    """
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.daemon = daemon
    thread.start()
    return thread


def format_stats(drop_count, xp_values, skills, avg_confidence):
    """
    Compose a status string summarizing drop count, XP, skills, and average confidence.
    """
    parts = [f"Drops: {drop_count}"]
    if xp_values:
        parts.append("XP: " + ", ".join(str(x) for x in xp_values))
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    parts.append(f"Conf: {avg_confidence:.2f}")
    return " | ".join(parts)
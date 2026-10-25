import os
import re
import time
from functools import lru_cache
from html import escape
from html.parser import HTMLParser

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()
app = Flask(__name__)

# Keep your API key in .env; never put it in frontend code.
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = None

if API_KEY:
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=35000),
    )
else:
    print("WARNING: GEMINI_API_KEY is missing; local fallback roadmaps will be used.")


class RoadmapHTMLSanitizer(HTMLParser):
    """Allow only simple formatting tags from the model; discard attributes."""
    ALLOWED = {"h3", "ul", "li", "strong", "p"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ALLOWED:
            self.parts.append(f"<{tag}>")

    def handle_endtag(self, tag):
        if tag in self.ALLOWED:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        self.parts.append(escape(data, quote=False))

    def get_html(self):
        return "".join(self.parts).strip()


def clean_and_sanitize(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    parser = RoadmapHTMLSanitizer()
    parser.feed(text)
    parser.close()
    return parser.get_html()


def extract_week_count(duration):
    match = re.search(r"(\d+)\s*weeks?", str(duration).lower())
    if match:
        return max(1, min(int(match.group(1)), 12))
    match = re.search(r"(\d+)\s*months?", str(duration).lower())
    if match:
        return max(1, min(int(match.group(1)) * 4, 12))
    return 4


def validate_roadmap(roadmap, weeks):
    if not roadmap:
        return False
    found = {
        int(n) for n in re.findall(
            r"<h3>\s*Week\s+(\d+)\b", roadmap, flags=re.IGNORECASE
        )
    }
    return set(range(1, weeks + 1)).issubset(found)


def get_topic_plan(goal):
    g = goal.lower()
    if "java" in g:
        titles = [
            ("Java setup and syntax", "Install/configure JDK and write Hello World.", "Run a basic Java program."),
            ("Variables and operators", "Practise data types, input, conversion, and operators.", "Write small calculation programs."),
            ("Conditions and loops", "Practise if/else, switch, for, and while.", "Solve decision and loop exercises."),
            ("Arrays and strings", "Practise traversal, searching, string methods, and StringBuilder.", "Solve array and string problems."),
            ("Methods and recursion", "Use parameters, return values, overloading, and basic recursion.", "Write reusable methods."),
            ("Object-oriented programming", "Learn classes, objects, constructors, and encapsulation.", "Build a class-based program."),
            ("Inheritance and interfaces", "Practise inheritance, overriding, interfaces, and abstraction.", "Design a small OOP application."),
            ("Exceptions and packages", "Use try/catch/finally, custom exceptions, and packages.", "Handle errors appropriately."),
            ("Collections framework", "Practise ArrayList, HashSet, HashMap, and iterators.", "Manage data with collections."),
            ("Files and generics", "Read/write files and practise generic types.", "Create a file-based utility."),
            ("Problem solving", "Solve mixed Java problems and plan a mini project.", "Complete core project features."),
            ("Project and revision", "Test, debug, document, and revise Java concepts.", "Demonstrate the finished project."),
        ]
    elif "python" in g:
        titles = [
            ("Python setup and syntax", "Install Python and practise print, input, and indentation.", "Run simple scripts."),
            ("Variables and data types", "Practise strings, numbers, booleans, and conversion.", "Use core data types."),
            ("Conditions and loops", "Use if/elif/else, for, while, and range.", "Solve beginner exercises."),
            ("Lists and tuples", "Practise indexing, slicing, and collection methods.", "Process sequences."),
            ("Dictionaries and sets", "Practise key-value data and set operations.", "Organize structured data."),
            ("Functions", "Use parameters, return values, and scope.", "Write reusable functions."),
            ("Modules and exceptions", "Use imports and try/except.", "Create a reliable script."),
            ("File handling", "Read/write text and CSV files.", "Build a file-based utility."),
            ("Object-oriented Python", "Learn classes, objects, and inheritance.", "Build an OOP program."),
            ("Useful libraries", "Explore a library relevant to your goal.", "Complete a practical exercise."),
            ("Mini project", "Build a task tracker or student record tool.", "Finish a working project."),
            ("Testing and revision", "Test the project and revise fundamentals.", "Explain your solution."),
        ]
    elif "dsa" in g or "data structure" in g or "algorithm" in g:
        titles = [
            ("Programming foundations", "Revise loops, functions, and arrays.", "Solve basic exercises."),
            ("Complexity analysis", "Learn time/space complexity and Big O.", "Analyse simple algorithms."),
            ("Arrays and strings", "Practise traversal, two pointers, and prefix sums.", "Solve array/string problems."),
            ("Linked lists", "Study singly and doubly linked lists.", "Implement list operations."),
            ("Stacks and queues", "Learn stack, queue, and deque operations.", "Solve related problems."),
            ("Hashing", "Use maps, sets, and frequency counting.", "Solve hashing problems."),
            ("Recursion", "Learn recursion and backtracking basics.", "Solve recursive exercises."),
            ("Searching and sorting", "Practise binary search and sorting algorithms.", "Compare algorithm complexity."),
            ("Trees", "Learn traversals and binary search trees.", "Solve tree problems."),
            ("Graphs", "Study BFS, DFS, and graph representation.", "Solve introductory graph tasks."),
            ("Dynamic programming", "Learn memoization and tabulation.", "Solve beginner DP problems."),
            ("Revision and practice", "Review patterns and solve mixed problems.", "Complete a timed practice set."),
        ]
    elif "web" in g or "frontend" in g or "full stack" in g:
        titles = [
            ("Web foundations", "Understand browsers, websites, and servers.", "Explain a basic web request."),
            ("HTML", "Practise semantic elements, forms, and structure.", "Create a structured page."),
            ("CSS foundations", "Learn selectors, box model, spacing, and typography.", "Style a webpage."),
            ("Responsive layouts", "Practise Flexbox, Grid, and media queries.", "Build a responsive layout."),
            ("JavaScript basics", "Learn variables, types, conditions, and loops.", "Write interactive scripts."),
            ("Functions and objects", "Practise functions, arrays, and objects.", "Solve JS exercises."),
            ("DOM and events", "Select elements and handle user events.", "Create an interactive page."),
            ("Forms and validation", "Build forms and validate input.", "Create a validated form."),
            ("APIs and JSON", "Learn fetch, HTTP basics, and JSON.", "Display API data."),
            ("Frontend project", "Combine HTML, CSS, JavaScript, and APIs.", "Build a functional frontend."),
            ("Testing and accessibility", "Check usability, responsiveness, and accessibility.", "Improve project quality."),
            ("Final project", "Polish, test, and document your project.", "Demonstrate the finished website."),
        ]
    else:
        titles = [
            ("Introduction and setup", f"Understand the foundations of {goal}.", "Explain the basic concepts."),
            ("Core fundamentals", f"Study essential concepts in {goal}.", "Complete beginner exercises."),
            ("Basic practical skills", f"Practise simple tasks related to {goal}.", "Apply the basics."),
            ("Problem solving", f"Solve beginner problems involving {goal}.", "Complete practical exercises."),
            ("Intermediate concepts", f"Explore intermediate concepts in {goal}.", "Apply new techniques."),
            ("Guided practice", f"Complete guided exercises using {goal}.", "Finish a practical task."),
            ("Tools and workflow", f"Learn tools used with {goal}.", "Use a tool in a small task."),
            ("Applied skills", f"Combine concepts from {goal}.", "Complete an applied exercise."),
            ("Advanced concepts", f"Explore more complex concepts in {goal}.", "Apply advanced ideas."),
            ("Project planning", f"Plan a small project using {goal}.", "Prepare a project outline."),
            ("Project development", f"Build and test a project related to {goal}.", "Create a working project."),
            ("Revision and presentation", f"Review {goal} and polish the project.", "Present your learning outcomes."),
        ]
    return titles


def create_fallback(goal, level, weeks):
    topics = get_topic_plan(goal)
    sections = []
    for week in range(1, weeks + 1):
        index = min(len(topics) - 1, ((week - 1) * len(topics)) // weeks)
        title, practice, outcome = topics[index]
        if level == "Advanced":
            level_note = "Extend this with optimization, testing, and challenging variations."
        elif level == "Intermediate":
            level_note = "Focus on implementation and moderately challenging problems."
        else:
            level_note = "Start with simple examples and practise each concept."
        sections.append(
            f"<h3>Week {week}: {escape(title)}</h3><ul>"
            f"<li><strong>Topics:</strong> {escape(title)} for {escape(goal)}.</li>"
            f"<li><strong>Practice:</strong> {escape(practice)} {escape(level_note)}</li>"
            f"<li><strong>Outcome:</strong> {escape(outcome)}</li></ul>"
        )
    return "\n".join(sections)


@lru_cache(maxsize=64)
def get_cached_roadmap(goal, level, duration, weeks):
    if client is None:
        raise RuntimeError("Gemini API key is not configured.")

    prompt = f"""Create a practical learning roadmap.
Goal: {goal}
Level: {level}
Duration: {duration}
Return exactly {weeks} sections, numbered Week 1 to Week {weeks}, with no skipped weeks.
For each week include one concise Topics item, one Practice item, and one Outcome item.
Make topics specific to the goal, progress from basics to a practical project, and suit the student's level.
Return HTML only, using only h3, ul, li, strong tags. No Markdown fences or introduction.
Format each section exactly like:
<h3>Week 1: Topic</h3><ul><li><strong>Topics:</strong> ...</li><li><strong>Practice:</strong> ...</li><li><strong>Outcome:</strong> ...</li></ul>
Keep each item short so the full roadmap fits in one response."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2400,
        ),
    )
    roadmap = clean_and_sanitize(response.text)
    if not validate_roadmap(roadmap, weeks):
        raise ValueError(f"AI returned an incomplete roadmap for {weeks} weeks.")
    return roadmap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-roadmap", methods=["POST"])
def generate_roadmap():
    data = request.get_json(silent=True) or {}
    goal = str(data.get("goal", "")).strip()[:150]
    level = str(data.get("level", "Beginner")).strip()
    duration = str(data.get("duration", "4 Weeks")).strip()

    if not goal:
        return jsonify({"success": False, "error": "Please enter what you want to learn."}), 400
    if level not in {"Beginner", "Intermediate", "Advanced"}:
        level = "Beginner"
    if duration not in {"4 Weeks", "8 Weeks", "12 Weeks"}:
        duration = "4 Weeks"

    weeks = extract_week_count(duration)
    started = time.monotonic()

    try:
        roadmap = get_cached_roadmap(goal, level, duration, weeks)
        elapsed = round(time.monotonic() - started, 2)
        print(f"AI roadmap ready in {elapsed}s (model={MODEL_NAME}, weeks={weeks}).")
        return jsonify({
            "success": True, "roadmap": roadmap, "weeks": weeks,
            "fallback": False, "elapsed_seconds": elapsed
        })
    except Exception as error:
        elapsed = round(time.monotonic() - started, 2)
        print(f"AI generation failed after {elapsed}s: {type(error).__name__}")
        fallback = create_fallback(goal, level, weeks)
        return jsonify({
            "success": True, "roadmap": fallback, "weeks": weeks,
            "fallback": True,
            "message": "AI is taking too long or is temporarily unavailable, so a ready-made roadmap is shown.",
            "elapsed_seconds": elapsed
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

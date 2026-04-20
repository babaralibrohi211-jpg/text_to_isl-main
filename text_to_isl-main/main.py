import json
import os
import stanza
import pprint
import ssl

from flask import Flask, request, render_template, send_from_directory
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree

# ---------------- SSL FIX ----------------
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder='static', static_url_path='')

# ---------------- BASE DIR ----------------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------- STANZA SETUP ----------------
STANZA_DIR = os.path.join(BASE_DIR, "stanza_resources")

# Download only once (safe for local setup)
if not os.path.exists(STANZA_DIR):
    stanza.download('en', model_dir=STANZA_DIR)

# FIXED PIPELINE (NO spaCy MIX)
en_nlp = stanza.Pipeline(
    'en',
    dir=STANZA_DIR,
    processors='tokenize,pos,lemma',
    use_gpu=False
)

# ---------------- STOP WORDS ----------------
stop_words = set([
    "am","are","is","was","were","be","being","been",
    "have","has","had","does","did","could","should",
    "would","can","shall","will","may","might","must","let"
])

# ---------------- GLOBAL VARIABLES ----------------
sent_list = []
sent_list_detailed = []

word_list = []
word_list_detailed = []

final_words = []
final_output_in_sent = []
final_words_dict = {}

# ---------------- STANFORD PARSER PATH ----------------
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser')
os.environ['STANFORD_MODELS'] = os.path.join(
    BASE_DIR,
    'stanford-parser/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
)

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET'])
def index():
    clear_all()
    return render_template('index.html')


@app.route('/', methods=['POST'])
def flask_test():
    clear_all()
    text = request.form.get('text')

    if not text:
        return ""

    take_input(text)

    for words in final_output_in_sent:
        for i, word in enumerate(words, start=1):
            final_words_dict[i] = word

    for key in final_words_dict:
        if len(final_words_dict[key]) == 1:
            final_words_dict[key] = final_words_dict[key].upper()

    return final_words_dict


# ---------------- INPUT PROCESSING ----------------
def take_input(text):
    text = text.strip().replace("\n", "").replace("\t", "")

    processed = ""
    if len(text) == 1:
        processed = text
    else:
        for word in text.split("."):
            processed += word.capitalize() + " ."

    doc = en_nlp(processed)
    convert(doc)


def convert(doc):
    convert_to_sentence_list(doc)
    convert_to_word_list(sent_list_detailed)

    for i, words in enumerate(word_list):
        word_list[i] = reorder_eng_to_isl(words)

    pre_process()
    convert_to_final()
    remove_punct(final_output_in_sent)


# ---------------- NLP HELPERS ----------------
def convert_to_sentence_list(doc):
    for sentence in doc.sentences:
        sent_list.append(sentence.text)
        sent_list_detailed.append(sentence)


def convert_to_word_list(sentences):
    for sentence in sentences:
        temp_list = []
        temp_detail = []

        for word in sentence.words:
            temp_list.append(word.text)
            temp_detail.append(word)

        word_list.append(temp_list)
        word_list_detailed.append(temp_detail)


def filter_words(words):
    return [w for w in words if w not in stop_words]


def remove_punct(word_list_data):
    for words, details in zip(word_list_data, word_list_detailed):
        for i, (w, d) in enumerate(zip(words, details)):
            if d.upos == "PUNCT":
                if w in words:
                    words.remove(w)


def pre_process():
    global final_words
    for words in word_list:
        final_words.append(filter_words(words))


# ---------------- PARSER ----------------
def reorder_eng_to_isl(input_string):
    try:
        parser = StanfordParser()
        trees = list(parser.parse(input_string))

        if not trees:
            return input_string

        parent_tree = ParentedTree.convert(trees[0])
        modified = modify_tree_structure(parent_tree)

        return modified.leaves()

    except Exception as e:
        print("Parser error:", e)
        return input_string


def modify_tree_structure(parent_tree):
    flag = {sub.treeposition(): 0 for sub in parent_tree.subtrees()}

    new_tree = ParentedTree('ROOT', [])
    i = 0

    for sub in parent_tree.subtrees():
        if sub.label() == "NP" and flag[sub.treeposition()] == 0:
            flag[sub.treeposition()] = 1
            new_tree.insert(i, sub)
            i += 1

        if sub.label() in ["VP", "PRP"]:
            for child in sub.subtrees():
                if child.label() in ["NP", "PRP"]:
                    if flag[child.treeposition()] == 0:
                        flag[child.treeposition()] = 1
                        new_tree.insert(i, child)
                        i += 1

    return new_tree


# ---------------- FINAL OUTPUT ----------------
def final_output(input_words):
    valid_words = set(open("words.txt", "r").read().split("\n"))
    result = []

    for word in input_words:
        word = word.lower()
        if word not in valid_words:
            result.extend(list(word))
        else:
            result.append(word)

    return result


def convert_to_final():
    for words in final_words:
        final_output_in_sent.append(final_output(words))


# ---------------- CLEAR ----------------
def clear_all():
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words.clear()
    final_output_in_sent.clear()
    final_words_dict.clear()


# ---------------- STATIC FILES ----------------
@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
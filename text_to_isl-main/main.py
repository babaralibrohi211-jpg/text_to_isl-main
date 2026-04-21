import json
import os
import spacy
from nltk.parse.stanford import StanfordParser
from nltk.tree import *
from six.moves import urllib
import zipfile
import sys
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__, static_folder='static', static_url_path='')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR,
                                             'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'

# Load spaCy small model (~50MB RAM vs stanza's ~600MB)
en_nlp = spacy.load('en_core_web_sm')

# ---------- Wrapper classes so rest of code works unchanged ----------

class WordWrapper:
    def __init__(self, token):
        self.text = token.text
        self.lemma = token.lemma_
        # spaCy uses 'PUNCT', stanza used 'PUNCT' via upos — same value
        self.upos = token.pos_

class SentenceWrapper:
    def __init__(self, span):
        self.text = span.text
        self.words = [WordWrapper(t) for t in span if not t.is_space]

class DocWrapper:
    def __init__(self, doc):
        self.sentences = [SentenceWrapper(s) for s in doc.sents]

# ---------- Stanford Parser helpers (unchanged) ----------

def is_parser_jar_file_present():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    return os.path.exists(stanford_parser_zip_file_path)

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.perf_counter()
        return
    duration = time.perf_counter() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    url = "https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip"
    urllib.request.urlretrieve(url, stanford_parser_zip_file_path, reporthook)

def extract_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    try:
        with zipfile.ZipFile(stanford_parser_zip_file_path) as z:
            z.extractall(path=BASE_DIR)
    except Exception:
        os.remove(stanford_parser_zip_file_path)
        download_parser_jar_file()
        extract_parser_jar_file()

def extract_models_jar_file():
    stanford_models_zip_file_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
    stanford_models_dir = os.environ.get('CLASSPATH')
    with zipfile.ZipFile(stanford_models_zip_file_path) as z:
        z.extractall(path=stanford_models_dir)

def download_required_packages():
    if not os.path.exists(os.environ.get('CLASSPATH')):
        if is_parser_jar_file_present():
            pass
        else:
            download_parser_jar_file()
        extract_parser_jar_file()
    if not os.path.exists(os.environ.get('STANFORD_MODELS')):
        extract_models_jar_file()

# ---------- NLP pipeline (unchanged logic) ----------

stop_words = set(["am", "are", "is", "was", "were", "be", "being", "been", "have", "has", "had",
                   "does", "did", "could", "should", "would", "can", "shall", "will", "may", "might", "must", "let"])

sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []


def convert_to_sentence_list(text):
    for sentence in text.sentences:
        sent_list.append(sentence.text)
        sent_list_detailed.append(sentence)


def convert_to_word_list(sentences):
    temp_list = []
    temp_list_detailed = []
    for sentence in sentences:
        for word in sentence.words:
            temp_list.append(word.text)
            temp_list_detailed.append(word)
        word_list.append(temp_list.copy())
        word_list_detailed.append(temp_list_detailed.copy())
        temp_list.clear()
        temp_list_detailed.clear()


def filter_words(word_list):
    temp_list = []
    final_words = []
    for words in word_list:
        temp_list.clear()
        for word in words:
            if word not in stop_words:
                temp_list.append(word)
        final_words.append(temp_list.copy())
    for words in word_list_detailed:
        for i, word in enumerate(words):
            if words[i].text in stop_words:
                del words[i]
                break
    return final_words


def remove_punct(word_list):
    for words, words_detailed in zip(word_list, word_list_detailed):
        for i, (word, word_detailed) in enumerate(zip(words, words_detailed)):
            if word_detailed.upos == 'PUNCT':
                del words_detailed[i]
                words.remove(word_detailed.text)
                break


def lemmatize(final_word_list):
    for words, final in zip(word_list_detailed, final_word_list):
        for i, (word, fin) in enumerate(zip(words, final)):
            if fin in word.text:
                if len(fin) == 1:
                    final[i] = fin
                else:
                    final[i] = word.lemma


def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}
    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag


def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i = i + 1
    return i, modified_parse_tree


def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i = i + 1
    return i, modified_parse_tree


def modify_tree_structure(parent_tree):
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i = i + 1
    return modified_parse_tree


def reorder_eng_to_isl(input_string):
    # If all single letters, no reordering needed
    count = 0
    for word in input_string:
        if len(word) == 1:
            count += 1
    if count == len(input_string):
        return input_string
    # Try Stanford Parser for ISL reordering, fall back to original order if unavailable
    try:
        download_required_packages()
        parser = StanfordParser()
        possible_parse_tree_list = [tree for tree in parser.parse(input_string)]
        parse_tree = possible_parse_tree_list[0]
        parent_tree = ParentedTree.convert(parse_tree)
        modified_parse_tree = modify_tree_structure(parent_tree)
        parsed_sent = modified_parse_tree.leaves()
        return parsed_sent
    except Exception as e:
        print(f"Stanford Parser unavailable, using original word order: {e}")
        return input_string


final_words = []
final_words_detailed = []


def pre_process(text):
    remove_punct(word_list)
    final_words.extend(filter_words(word_list))
    lemmatize(final_words)


def final_output(input):
    final_string = ""
    valid_words = open("words.txt", 'r').read()
    valid_words = valid_words.split('\n')
    fin_words = []
    for word in input:
        word = word.lower()
        if word not in valid_words:
            for letter in word:
                fin_words.append(letter)
        else:
            fin_words.append(word)
    return fin_words


final_output_in_sent = []


def convert_to_final():
    for words in final_words:
        final_output_in_sent.append(final_output(words))


def take_input(text):
    test_input = text.strip().replace("\n", "").replace("\t", "")
    test_input2 = ""
    if len(test_input) == 1:
        test_input2 = test_input
    else:
        for word in test_input.split("."):
            test_input2 += word.capitalize() + " ."

    # Use spaCy via wrapper instead of stanza
    doc = en_nlp(test_input2)
    some_text = DocWrapper(doc)
    convert(some_text)


def convert(some_text):
    convert_to_sentence_list(some_text)
    convert_to_word_list(sent_list_detailed)
    for i, words in enumerate(word_list):
        word_list[i] = reorder_eng_to_isl(words)
    pre_process(some_text)
    convert_to_final()
    remove_punct(final_output_in_sent)


def clear_all():
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words.clear()
    final_words_detailed.clear()
    final_output_in_sent.clear()
    final_words_dict.clear()


final_words_dict = {}


@app.route('/', methods=['GET'])
def index():
    clear_all()
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def flask_test():
    clear_all()
    text = request.form.get('text')
    print("text is", text)
    if text == "":
        return ""
    take_input(text)

    for words in final_output_in_sent:
        for i, word in enumerate(words, start=1):
            final_words_dict[i] = word

    for key in final_words_dict.keys():
        if len(final_words_dict[key]) == 1:
            final_words_dict[key] = final_words_dict[key].upper()

    print(final_words_dict)
    return final_words_dict


@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
import random
import webbrowser
import os
import ast
from bs4 import BeautifulSoup
from data import DATA_DIR

VISUAL_FILE_NAME = 'visualization.html'
current_directory = os.getcwd()


def random_color():
    alpha = "a3"
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) + alpha


def get_compared(parsed_doc, parsed_preds):
    ground_truth_clusters = parsed_doc.find_all("tc:entity")

    # Prepare dictionaries for predictions
    color_by_cluster_id = {}
    color_by_mention_prediction = {}
    for mention_id, cluster_id in parsed_preds.items():
        if cluster_id not in color_by_cluster_id:
            color_by_cluster_id[cluster_id] = random_color()

        color = color_by_cluster_id[cluster_id]
        color_by_mention_prediction[mention_id] = color

    # Get dictionary of tokens
    token_by_id = {}
    for token in parsed_doc.find_all("tc:token"):
        token_by_id[token['id']] = token.text

    # Get dictionary of colors for each token and tokens for mentions
    cluster_color_by_token = {}
    mention_by_token = {}
    for cluster in ground_truth_clusters:
        color = random_color()
        for reference in cluster.find_all("tc:reference"):
            rc_id = reference['id']
            for token in reference['tokenids'].split(" "):
                cluster_color_by_token[token] = color
                mention_by_token[token] = rc_id

    # Combine to text
    ground_truth = ""
    predictions = ""
    for sentence in parsed_doc.find_all("tc:sentence"):
        if ground_truth != "":
            ground_truth += "<br><br>"
            predictions += "<br><br>"

        row_truth = ""
        row_predictions = ""
        for token in sentence["tokenids"].split(" "):
            if row_truth != "":
                row_truth += " "
                row_predictions += " "

            # Ground truth logic
            if token in cluster_color_by_token:
                color = cluster_color_by_token[token]
                row_truth += f"""<span style="background-color:{color}">"""+token_by_id[token]+"</span>"
            else:
                row_truth += token_by_id[token]

            # Predictions
            if token in mention_by_token:
                mention = mention_by_token[token]
                if mention in color_by_mention_prediction:
                    color = color_by_mention_prediction[mention]
                    row_predictions += f"""<span style="background-color:{color}">"""+token_by_id[token]+"</span>"
            else:
                row_predictions += token_by_id[token]

        ground_truth += row_truth
        predictions += row_predictions

    return f"""
    <div style="width:100%;display:flex">
        <div style="width:45%;"><h3>Ground truth</h3>{ground_truth}</div>
        <div style="width:10%;"></div>
        <div style="width:45%;"><h3>Model predictions</h3>{predictions}</div>
    </div>
    """


def parse_document(document_name):
    file_path = os.path.join(DATA_DIR, document_name+".tcf")
    with open(file_path, encoding="utf8") as f:
        content = f.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    return soup


def parse_predictions(clusters):
    return ast.literal_eval(clusters)

def get_document_predictions(test_preds_file):
    ul_elements = ""
    document_content = ""

    with open(test_preds_file) as f:
        predictions = f.readlines()[5:]

    for i in range(0, len(predictions), 2):
        document = predictions[i].split('Document ')[1].split(':')[0].split('\\')[1].replace("'", "")
        clusters = predictions[i+1]

        parsed_doc = parse_document(document)
        parsed_preds = parse_predictions(clusters)

        text = get_compared(parsed_doc, parsed_preds)

        tab_id = "id" + str(i)
        ul_elements += f"""<li class="nav-item"><a class="nav-link" href="#{tab_id}" data-toggle="tab">{document}</a></li>"""
        document_content += f"""<div class="tab-pane" id="{tab_id}" style="width:100%">{text}</div>"""

    predictions = f"""
    <div>
        <ul class="nav nav-tabs">
            {ul_elements}
        </ul>
        
        <div class="tab-content" style="margin-top:30px; margin-bottom:150px">
            {document_content}
        </div>
    </div>
    """

    return predictions


def get_test_scores(test_preds_file):
    # Read first 4 lines from test file
    with open(test_preds_file) as f:
        test_scores_txt = [f.readline() for x in range(4)]

    test_scores_txt = "<br>".join(test_scores_txt)
    return f"""
        <div style="margin:20px">
            {test_scores_txt}
        </div>
    """


def write_body(visual_path, test_preds_file):
    test_scores = get_test_scores(test_preds_file)
    document_predictions = get_document_predictions(test_preds_file)

    body = f"""
        <body>
            <h2>Visualization of predictions</h2>
            {test_scores}
            {document_predictions}
        </body>
    """
    with open(visual_path, "a", encoding="utf8") as f:
        f.write(body)


def write_footer(visual_path):
    footer = f"""</html>"""
    with open(visual_path, "a") as f:
        f.write(footer)


def write_header(visual_path):
    css_file_path = os.path.join(current_directory, 'visualization', 'bootstrap.min.css')
    header = f"""
        <html>
            <head>
                <title>Visualization report</title>
                <link rel="stylesheet" type="text/css" href="file:///{css_file_path}">
                <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
                <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js" integrity="sha384-wfSDF2E50Y2D1uUdj0O3uMBJnjuUD4Ih7YwaYd1iqfktj0Uod8GCExl3Og8ifwB6" crossorigin="anonymous"></script>
                <style>
                    body {{
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    align-items: center;
                    margin: 60px;
                    }}
                </style>
            </head>
    """

    with open(visual_path, "a") as f:
        f.write(header)


def build_visualization(test_preds_file, save_dir):
    # Set visualization file path
    visual_path = os.path.join(save_dir, VISUAL_FILE_NAME)

    # Initialize visualization file
    open(visual_path, 'w').close()

    # Write all parts of visualization
    write_header(visual_path)
    write_body(visual_path, test_preds_file)
    write_footer(visual_path)

    return visual_path


# Open visualization file in browser (in a new tab, if possible)
def display_visualization(url):
    webbrowser.open("file://"+url, new=2)


def build_and_display(test_preds_file, save_dir, display):
    """
    Will build, save and display visualization in browser
    :param test_preds_file: absolute path to test files
    :param save_dir: absolute path to directory to save visualization
    :param display: will only generate visualization if set to false
    """
    test_preds_file = os.path.join(current_directory, test_preds_file)
    save_dir = os.path.join(current_directory, save_dir)
    visual_path = build_visualization(test_preds_file, save_dir)

    if display:
        display_visualization(visual_path)


# Only for testing
if __name__ == "__main__":
    build_and_display("./test_preds.txt", current_directory, True)

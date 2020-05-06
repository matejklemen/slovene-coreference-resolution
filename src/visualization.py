import random
import webbrowser
import os
import ast
from bs4 import BeautifulSoup
from data import COREF149_DIR, SENTICOREF_DIR, read_senticoref_doc

VISUAL_FILE_NAME = 'visualization.html'
current_directory = os.getcwd()


def random_color():
    alpha = "a3"
    return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) + alpha


def get_compared(parsed_doc, parsed_preds, doc_id):
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
    gt_color_ids = {}
    preds_color_ids = {}
    unique_classes = {}
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
                color_id = len(gt_color_ids)
                if color in gt_color_ids:
                    color_id = gt_color_ids[color]
                else:
                    gt_color_ids[color] = color_id
                cls = f"""{doc_id}-gt-c-{color_id}"""
                unique_classes[cls] = 1
                row_truth += f"""<span style="background-color:{color};border-radius: 3px;padding: 0px 2px;" class="{cls}">"""+token_by_id[token]+"</span>"
            else:
                row_truth += token_by_id[token]

            # Predictions
            if token in mention_by_token:
                mention = mention_by_token[token]
                if mention in color_by_mention_prediction:
                    color = color_by_mention_prediction[mention]
                    color_id = len(preds_color_ids)
                    if color in preds_color_ids:
                        color_id = preds_color_ids[color]
                    else:
                        preds_color_ids[color] = color_id
                    cls = f"""{doc_id}-pr-c-{color_id}"""
                    unique_classes[cls] = 1
                    row_predictions += f"""<span style="background-color:{color};border-radius: 3px;padding: 0px 2px;" class="{cls}">"""+token_by_id[token]+"</span>"
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
    <script>
        var allClassNames =  [{', '.join('"{0}"'.format(c) for c in unique_classes.keys())}];

        allClassNames.forEach(function(clsName) {{
            allClasses = document.getElementsByClassName(clsName);
            
            for (var i = 0; i < allClasses.length; i++) {{
                cls = allClasses[i];
                cls.addEventListener("mouseenter", (e) => {{
                    var bgColor = $('.'+clsName).css( 'background-color');
                    $('.'+clsName).attr('data-color', bgColor);
                    $('.'+clsName).css( 'background-color', 'black');
                    $('.'+clsName).css( 'color', 'white');
                }});

                cls.addEventListener("mouseleave", (e) => {{
                    var bgColor = $('.'+clsName).attr('data-color')
                    $('.'+clsName).css( 'background-color', bgColor);
                    $('.'+clsName).css( 'color', 'initial');
                }});
            }}
        }})
        
    </script>
    """


def get_compared_senticoref(parsed_doc, parsed_preds, doc_id):

    # Prepare dictionaries for predictions
    color_by_cluster_id = {}
    color_by_mention_prediction = {}
    for mention_id, cluster_id in parsed_preds.items():
        if cluster_id not in color_by_cluster_id:
            color_by_cluster_id[cluster_id] = random_color()

        color = color_by_cluster_id[cluster_id]
        color_by_mention_prediction[mention_id] = color

    # Get dictionary of colors for each token and tokens for mentions
    cluster_color_by_token = {}
    mention_by_token = {}
    for cluster in parsed_doc.clusters:
        color = random_color()
        for rc_id in cluster:
            for token in parsed_doc.mentions[rc_id].tokens:
                cluster_color_by_token[token.token_id] = color
                mention_by_token[token.token_id] = rc_id

    # Combine to text
    ground_truth = ""
    predictions = ""
    gt_color_ids = {}
    preds_color_ids = {}
    unique_classes = {}
    for sentence in parsed_doc.sents:
        if ground_truth != "":
            ground_truth += "<br><br>"
            predictions += "<br><br>"

        row_truth = ""
        row_predictions = ""
        for token_id in sentence:
            if row_truth != "":
                row_truth += " "
                row_predictions += " "

            # Ground truth logic
            if token_id in cluster_color_by_token:
                color = cluster_color_by_token[token_id]
                color_id = len(gt_color_ids)
                if color in gt_color_ids:
                    color_id = gt_color_ids[color]
                else:
                    gt_color_ids[color] = color_id
                cls = f"""{doc_id}-gt-c-{color_id}"""
                unique_classes[cls] = 1
                row_truth += f"""<span style="background-color:{color};border-radius: 3px;padding: 0px 2px;" class="{cls}">""" + \
                             parsed_doc.tokens[token_id].raw_text + "</span>"
            else:
                row_truth += parsed_doc.tokens[token_id].raw_text

            # Predictions
            if token_id in mention_by_token:
                mention = mention_by_token[token_id]
                if mention in color_by_mention_prediction:
                    color = color_by_mention_prediction[mention]
                    color_id = len(preds_color_ids)
                    if color in preds_color_ids:
                        color_id = preds_color_ids[color]
                    else:
                        preds_color_ids[color] = color_id
                    cls = f"""{doc_id}-pr-c-{color_id}"""
                    unique_classes[cls] = 1
                    row_predictions += f"""<span style="background-color:{color};border-radius: 3px;padding: 0px 2px;" class="{cls}">""" + \
                                       parsed_doc.tokens[token_id].raw_text + "</span>"
            else:
                row_predictions += parsed_doc.tokens[token_id].raw_text

        ground_truth += row_truth
        predictions += row_predictions

    return f"""
        <div style="width:100%;display:flex">
            <div style="width:45%;"><h3>Ground truth</h3>{ground_truth}</div>
            <div style="width:10%;"></div>
            <div style="width:45%;"><h3>Model predictions</h3>{predictions}</div>
        </div>
        <script>
            var allClassNames =  [{', '.join('"{0}"'.format(c) for c in unique_classes.keys())}];
    
            allClassNames.forEach(function(clsName) {{
                allClasses = document.getElementsByClassName(clsName);
                
                for (var i = 0; i < allClasses.length; i++) {{
                    cls = allClasses[i];
                    cls.addEventListener("mouseenter", (e) => {{
                        var bgColor = $('.'+clsName).css( 'background-color');
                        $('.'+clsName).attr('data-color', bgColor);
                        $('.'+clsName).css( 'background-color', 'black');
                        $('.'+clsName).css( 'color', 'white');
                    }});
    
                    cls.addEventListener("mouseleave", (e) => {{
                        var bgColor = $('.'+clsName).attr('data-color')
                        $('.'+clsName).css( 'background-color', bgColor);
                        $('.'+clsName).css( 'color', 'initial');
                    }});
                }}
            }})
        </script>
        """

def parse_document_senticoref(document_name):
    file_path = os.path.join(SENTICOREF_DIR, document_name + ".tsv")
    return read_senticoref_doc(file_path)


def parse_document(document_name):
    file_path = os.path.join(COREF149_DIR, document_name+".tcf")
    with open(file_path, encoding="utf8") as f:
        content = f.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml")
    return soup


def parse_predictions(clusters):
    return ast.literal_eval(clusters)


def get_document_predictions(test_preds_file, database_name):
    ul_elements = ""
    document_content = ""

    with open(test_preds_file) as f:
        # skip 1st line
        predictions = f.readlines()[1:]

    for i in range(0, len(predictions), 2):
        doc_line = predictions[i]
        document = ""
        try:
            striped = doc_line.split('Document ')[1].split(':')[0]
            document = striped.split('\\')
            if len(document) > 1:
                document = document[1]
            else:
                document = document[0]
            document = document.replace("'", "")
        except:
            print("Document name could not be obtained. Tried parsing line: " + doc_line)

        clusters = predictions[i+1]

        parsed_preds = parse_predictions(clusters)

        if database_name == 'coref149':
            parsed_doc = parse_document(document)
            text = get_compared(parsed_doc, parsed_preds, i)
        else:
            parsed_doc = parse_document_senticoref(document)
            text = get_compared_senticoref(parsed_doc, parsed_preds, i)


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


def get_test_scores(pred_scores_file):
    # Read lines from pred_scores file
    with open(pred_scores_file) as f:
        scores_txt = f.readlines()

    # pred_scores file can contain scores for multiple models.
    # find all lines starting with "Test scores" and print them in separate

    score_indexes = []
    for index, line in enumerate(scores_txt):
        if line.startswith("Test scores"):
            score_indexes.append(index)

    before = """
    <div class="container" style="margin:20px">
        <div class="row">
    """
    after = """
        </div>
    </div>
    """

    scores = ""
    for line_index in score_indexes:
        txt = ""
        txt += """<div class="col">"""
        txt += "<br>".join(scores_txt[line_index:line_index+5])
        txt += """</div>"""
        scores += txt

    return before + scores + after


def get_database_name(pred_scores_file):
    with open(pred_scores_file) as f:
        return f.readlines()[0].split('Database: ')[1].rstrip()


def write_body(visual_path, pred_clusters_file, pred_scores_file):
    test_scores = get_test_scores(pred_scores_file)
    database_name = get_database_name(pred_scores_file)
    # Remove database name (will be in title)
    document_predictions = get_document_predictions(pred_clusters_file, database_name)

    body = f"""
        <body>
            <h2>Visualization of predictions for {database_name}</h2>
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
    header = f"""
        <html>
            <head>
                <title>Visualization report</title>
                <link rel="stylesheet" type="text/css" href="https://bootswatch.com/4/litera/bootstrap.min.css">
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


def build_visualization(pred_clusters_file, pred_scores_file, save_dir):
    # Set visualization file path
    visual_path = os.path.join(save_dir, VISUAL_FILE_NAME)

    # Initialize visualization file
    open(visual_path, 'w').close()

    # Write all parts of visualization
    write_header(visual_path)
    write_body(visual_path, pred_clusters_file, pred_scores_file)
    write_footer(visual_path)

    return visual_path


# Open visualization file in browser (in a new tab, if possible)
def display_visualization(url):
    webbrowser.open("file://"+url, new=2)


def build_and_display(pred_clusters_file, pred_scores_file, save_dir, display):
    """
    Will build, save and display visualization in browser
    :param pred_clusters_file: absolute path to test files
    :param pred_scores_file: absolute path to test files
    :param save_dir: absolute path to directory to save visualization
    :param display: will only generate visualization if set to false
    """
    pred_clusters_file = os.path.join(current_directory, pred_clusters_file)
    pred_scores_file = os.path.join(current_directory, pred_scores_file)
    save_dir = os.path.join(current_directory, save_dir)
    visual_path = build_visualization(pred_clusters_file, pred_scores_file, save_dir)

    if display:
        display_visualization(visual_path)


# Only for testing
if __name__ == "__main__":
    build_and_display("./pred_clusters.txt", "./pred_scores.txt", current_directory, True)

import webbrowser
import os

VISUAL_FILE_NAME = 'visualization.html'
current_directory = os.getcwd()


def get_document_predictions(test_preds_file):
    ul_elements = ""
    document_content = ""

    with open(test_preds_file) as f:
        predictions = f.readlines()[6:]

    for i in range(0, len(predictions), 2):
        document = predictions[i].split('Document ')[1]
        clusters = predictions[i+1]

        tab_id = "id" + str(i)
        ul_elements += f"""<li class="nav-item"><a class="nav-link" href="#{tab_id}" data-toggle="tab">{document}</a></li>"""
        document_content += f"""<div class="tab-pane active" id="{tab_id}">{clusters}</div>"""

    predictions = f"""
    <div>
        <ul class="nav nav-tabs">
            {ul_elements}
        </ul>
        
        <div class="tab-content">
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
    with open(visual_path, "a") as f:
        f.write(body)


def write_footer(visual_path):
    footer = f"""</html>"""
    with open(visual_path, "a") as f:
        f.write(footer)


def write_header(visual_path):
    css_file_path = os.path.join(current_directory, 'visualization', 'bootstrap.min.css')
    header = f"""
        <html>
            <header>
                <title>Visualization report</title>
                <link rel="stylesheet" type="text/css" href="{css_file_path}">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
                <style>
                    body {{
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    align-items: center;
                    margin: 60px;
                    }}
                </style>
            </header>
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
    print("url: ", url)
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
    print("args: ", test_preds_file , " ", save_dir)
    visual_path = build_visualization(test_preds_file, save_dir)

    if display:
        display_visualization(visual_path)


# Only for testing
if __name__ == "__main__":
    build_and_display("./test_preds.txt", current_directory, true)

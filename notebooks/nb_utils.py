from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np



def draw_heatmap(matrix, exp=0.1, title='Word Frequency Heatmap', cluster=True, labels=None):

    if cluster:
        # Sort the image with DBSCAN 
        x_scaled = StandardScaler().fit_transform(matrix)
        similarity_matrix = cosine_similarity(x_scaled)

        # PARAMETER: 5% of rows
        MIN_SAMPLES = max(7, int( similarity_matrix.shape[0]*0.01 + 1 ))
        clustering = DBSCAN(eps=2.5, min_samples=MIN_SAMPLES).fit(similarity_matrix)
        sorted_indices = np.argsort(clustering.labels_)
        # st.write(sorted_indices)
        matrix = matrix[sorted_indices]
        if not labels:
            labels = clustering.labels_[sorted_indices]

    if len(matrix) == 1:
        figY = 0.5
    else:
        figY = min( 5, max(2, 0.5 * len(matrix) ) )
    fig, ax = plt.subplots(figsize=(12, figY))

    # Plot the heatmap
    cax = ax.matshow(np.power(matrix, exp), cmap='Blues', aspect='auto')

    # Create a colormap for the labels
    unique_labels = np.unique(labels)
    num_colors = len(unique_labels)
    colors = plt.cm.get_cmap('tab20', num_colors)
    row_colors = [colors(labels[i]) for i in range(len(labels))]

    # Apply row colors
    for idx, label in enumerate(labels):
        ax.add_patch(plt.Rectangle((0, idx), matrix.shape[1], 1, color=row_colors[idx], alpha=0.2, lw=0))

    # _ = ax.imshow(np.power(matrix, exp), cmap='Blues', aspect='auto')
    ax.set_xlabel('Word Index')
    ax.set_yticks([])
    ax.set_title(title)

    return fig


def parse_docstring(docstring):
    if not docstring:
        return None
    lines = docstring.split('\n')
    if len(lines) > 0 and lines[0].strip() != "":
        return lines[0].strip()
    elif len(lines) > 1 and lines[1].strip() != "":
        return lines[1].strip()
    else:
        return "Documentation pending"

def showAttribs(classObj):
    attribs = [att for att in dir(classObj) if not att.startswith('_')]
    properties = [att for att in attribs if not callable(getattr(classObj, att))]
    methods = [att for att in attribs if callable(getattr(classObj, att))]
   
    doc = parse_docstring(classObj.__doc__)
    print(f"{classObj.__name__:20}: {doc}")

    print('\nConstructor\n================')
    doc = parse_docstring(getattr(classObj, '__init__').__doc__)
    print(f"__init__            : {doc}")

    print('\nProperties\n================')
    for prop in properties:
        doc = parse_docstring(getattr(classObj, prop).__doc__)
        print(f"{prop:20}: {doc}")

    print('\nMethods\n================')
    for method in methods:
        doc = parse_docstring(getattr(classObj, method).__doc__)    
        print(f"{method:20}: {doc}")
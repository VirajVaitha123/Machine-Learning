import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

#evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

#Visualisations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def multi_eval(model,X,y_acutal,y_pred,cv=3):
        """
        Determine important evaluation metrics of multiclass classification.
        For example:
            * ``average precision`` will be the average precision across very class
            * ``average recall`` will be the average recall across every class
            ``confusion matrix`` can help debug and truly understand your model is performing.
        
        Parameters
        ----------
        model : model capable of accepting features and returning an array of predictions
        X: training, test or validation features | array-like
        y_actual: actual values (label/target column) accompying the X array | array-like
        y_pred: prediction values (label/target colum) from the model | array-like
        Returns
        -------
        target_type : float or array-like
            *``accuracy, precision or recall will output a float value``
            * confusion matrix will output an array 
            
        Examples
        --------
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        log_reg = LogisticRegression(random_state =42,
                                multi_class = "auto",
                                solver ='liblinear')


        log_reg.fit(X_train, y_train)
    
        metrics(log_reg,X_train,y_train,y_train_pred,cv=3)
        
        >>> [[133   0   0   0   0   1   0   0   1   0]
        [  0 129   1   2   0   0   0   0   9   4]
        [  0   0 136   3   0   0   0   0   0   0]
        [  0   0   0 125   0   2   0   4   3   3]
        [  0   3   0   0 121   0   0   1   1   0]
        [  0   1   0   1   0 117   0   2   1   1]
        [  0   1   0   0   1   0 133   0   1   0]
        [  0   1   0   1   1   0   0 133   1   1]
        [  0   6   0   1   1   1   0   1 125   1]
        [  0   0   0   1   0   2   0   2   2 125]]
        >>> Accuracy  = 94.51%
        >>> Average Precision = 94.92%
        >>> Average Recall = 94.85%
        
        """
        multi_eval.accuracy = cross_val_score(model,X,y_acutal)
        multi_eval.conf_mx = confusion_matrix(y_acutal, y_pred)
        
        multi_eval.recall = np.diag(multi_eval.conf_mx) / np.sum(multi_eval.conf_mx, axis = 1)
        multi_eval.recall = np.mean(multi_eval.recall)
        
        multi_eval.precision = np.diag(multi_eval.conf_mx) / np.sum(multi_eval.conf_mx, axis = 0)
        multi_eval.precision = np.mean(multi_eval.precision)
        print(multi_eval.conf_mx)
        print("Accuracy  = " + str(round(np.mean(multi_eval.accuracy)*100,2)) + "%")
        print("Average Precision = " + str(round(multi_eval.precision*100,2)) + "%")
        print("Average Recall = " + str(round(multi_eval.recall*100,2)) + "%")

def multi_confmxs(confmx, row_labels, col_labels,accuracy =None,precision =None,recall =None, normalised_errors = False):
    """
    Heat map to highlight the correct predictions and optional paramters to print other important classification metrics.
    
       
    Parameters
    ----------
    confmx : confusion matrix showing absolute correct predictions | array-like
    accuracy : accuracy | integer or float value 
    precision : precision | integer or float value 
    recall : recall | integer or float value 
    -------
    target_type : heatmap visulisation and printed metrics
    
    Examples
    --------
    """
    if normalised_errors == False:
        fig, ax = plt.subplots()
        im, cbar = heatmap(confmx ,row_labels, col_labels, ax=ax,
        cmap=plt.cm.PuRd, cbarlabel="Count")
        
        texts = annotate_heatmap(im, valfmt="{x:.0f}")
        title = plt.title('Confusion Matrix')
        title.set_position([.5, 1.15])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.gca().xaxis.tick_bottom()
        plt.show()
        print("hello")
        if isinstance(np.mean(accuracy), int) or isinstance(np.mean(accuracy), float):
            print("Accuracy  = " + str(round(np.mean(accuracy)*100,2)) + "%")
        else: 
            pass
        if isinstance(precision, int) or isinstance(precision, float):
            print("Average Precision = " + str(round(precision*100,2)) + "%")
        else:
            pass
        if isinstance(precision, int) or isinstance(precision, float):
            print("Average Recall = " + str(round(recall*100,2)) + "%")
        else:
            pass
    
        

    else:
        row_sums = confmx.sum(axis=1, keepdims=True) #Total Images per number (i.e total number of 1,2,3,4,5,6,7,8,9,10)
        norm_conf_mx = confmx / row_sums #Errors as a % per class (i.e 5% of 8's are incorrectly classified as a 5. This will help us analyze errors more effectively)
        np.fill_diagonal(norm_conf_mx,0)  #Remove correct predictions (Diagonals as when we predicted a value correctly)
        fig, ax = plt.subplots()
        im, cbar = heatmap(norm_conf_mx ,row_labels, col_labels, ax=ax,
        cmap=plt.cm.PuRd, cbarlabel="% of errors")
        
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        title = plt.title('Confusion Matrix highlighting normalised errors')
        title.set_position([.5, 1.15])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.gca().xaxis.tick_bottom()
        plt.show()
        
        if isinstance(np.mean(accuracy), int) or isinstance(np.mean(accuracy), float):
            print("Accuracy  = " + str(round(np.mean(accuracy)*100,2)) + "%")
        else: 
            pass
        if isinstance(precision, int) or isinstance(precision, float):
            print("Average Precision = " + str(round(precision*100,2)) + "%")
        else:
            pass
        if isinstance(recall, int) or isinstance(recall, float):
            print("Average Recall = " + str(round(recall*100,2)) + "%")
        else:
            pass
        






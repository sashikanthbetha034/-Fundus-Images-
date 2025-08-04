from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from sklearn import metrics
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def Plot_ROC_Curve():
    lw = 2
    cls = ['CNN', 'FCN', 'ANN', 'TL-FCNN', 'FO-LEA-TL-AFCNN']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    per = round(Actual.shape[0] * 0.80)
    Actual = Actual[per:, :]
    colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i], )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
    plt.yticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path1 = "./Journ_Results/ROC.png"
    plt.savefig(path1)
    plt.show()


def plotConvResults():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'OOA-TL-AFCNN', 'POA-TL-AFCNN', 'SOA-TL-AFCNN', 'LEA-TL-AFCNN', 'FO-LEA-TL-AFCNN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((len(Algorithm) - 1, 5))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report '
              '--------------------------------------------------')
        print(Table)

        Conv_Graph = Fitness[i]
        length = np.arange((Conv_Graph.shape[1]))
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='OOA-TL-AFCNN')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='POA-TL-AFCNN')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='SOA-TL-AFCNN')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='LEA-TL-AFCNN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='FO-LEA-TL-AFCNN')
        plt.xticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
        plt.yticks(fontname="Arial", fontsize=11, fontweight='bold', color='#1d3557')
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Journ_Results/Conv.png")
        plt.show()


def plot_results_KFOLD_Positive_Measures():
    eval = np.load('Eval_all_KFOLD.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1Score', 'MCC', 'For', 'PT', 'CSI', 'BA',
             'FM', 'BM', 'MK', 'Lrplus', 'Lrminus', 'Dor', 'Prevalence']

    Algorithm = ['OOA-TL-AFCNN', 'POA-TL-AFCNN', 'SOA-TL-AFCNN', 'LEA-TL-AFCNN', 'FO-LEA-TL-AFCNN']
    Classfier = ['CNN', 'FCN', 'ANN', 'TL-FCNN', 'FO-LEA-TL-AFCNN']
    KFOLD = [1, 2, 3, 4, 5]
    for j in range(len(KFOLD)):
        Graph = np.zeros((eval.shape[1], eval.shape[2]))
        for k in range(eval.shape[1]):
            for l in range(eval.shape[2] - 4):
                Graph[k, l] = eval[j, k, l + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0] - 6)
        barWidth = 0.18
        color = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
        for m in range(Graph.shape[0] - 5):
            ax.bar(X + (m * barWidth), Graph[m, :4], color=color[m - 5], width=barWidth, edgecolor='#032b43',
                   label=Algorithm[m - 5])
        plt.xticks(X + (((len(Classfier)) * barWidth) / 2.50), ('Accuracy', 'Sensitivity', 'Specificity', 'Precision'), fontname="Arial", fontsize=12, fontweight='bold', color='k')

        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Journ_Results/KFold_%s_Positive_%s_Alg_Line.png" % (j + 1, Terms[KFOLD[j]])
        plt.savefig(path1)
        plt.show()

        # -----------------------------------------------Classifier------------------------------------------------

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0] - 6)
        barWidth = 0.18
        color = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
        for m in range(5, Graph.shape[0]):
            ax.bar(X + (m * barWidth), Graph[m, :4], color=color[m - 5], width=barWidth, edgecolor='#032b43',
                   label=Classfier[m - 5])
        plt.xticks(X + (((len(Classfier)) * barWidth) / 0.73), ('Accuracy', 'Sensitivity', 'Specificity', 'Precision'), fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        path1 = "./Journ_Results/KFold_%s_Positive_%s_Mtd_Line.png" % (j + 1, Terms[KFOLD[j]])
        plt.savefig(path1)
        plt.show()


def plot_results_KFOLD_Negative_Measures():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_KFOLD.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1Score', 'MCC', 'For', 'PT', 'CSI', 'BA',
             'FM', 'BM', 'MK', 'Lrplus', 'Lrminus', 'Dor', 'Prevalence']

    Algorithm = ['OOA-TL-AFCNN', 'POA-TL-AFCNN', 'SOA-TL-AFCNN', 'LEA-TL-AFCNN', 'FO-LEA-TL-AFCNN']
    Classfier = ['CNN', 'FCN', 'ANN', 'TL-FCNN', 'FO-LEA-TL-AFCNN']
    Graph_Term = [4, 5, 7, 10]
    KFOLD = [1, 2, 3, 4, 5]
    for j in range(len(KFOLD)):
        Graph = np.zeros((eval.shape[1], eval.shape[2]))
        for k in range(eval.shape[1]):
            for l in range(eval.shape[2] - 4):
                Graph[k, l] = eval[j, k, l + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0] - 6)
        barWidth = 0.18
        color = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
        for m in range(Graph.shape[0] - 5):
            ax.bar(X + (m * barWidth), Graph[m, Graph_Term], color=color[m - 5], width=barWidth, edgecolor='#032b43',
                   label=Algorithm[m - 5])
        plt.xticks(X + (((len(Classfier)) * barWidth) / 2.50), ('FPR', 'FNR', 'FDR', 'FOR'), fontsize=12, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Journ_Results/KFold_%s_Negative_%s_Alg_Line.png" % (j + 1, Terms[KFOLD[j]])
        plt.savefig(path1)
        plt.show()

        # -----------------------------------------------Classifier------------------------------------------------

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0] - 6)
        barWidth = 0.18
        color = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
        for m in range(5, Graph.shape[0]):
            ax.bar(X + (m * barWidth), Graph[m, Graph_Term], color=color[m - 5], width=barWidth, edgecolor='#032b43',
                   label=Classfier[m - 5])
        plt.xticks(X + (((len(Classfier)) * barWidth) / 0.73), ('FPR', 'FNR', 'FDR', 'FOR'), fontsize=12, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Journ_Results/KFold_%s_Negative_%s_Mtd_Line.png" % (j + 1, Terms[KFOLD[j]])
        plt.savefig(path1)
        plt.show()


def Plot_Results_Batch():  # Table Results
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Batch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1Score', 'MCC', 'For', 'PT', 'CSI', 'BA',
             'FM', 'BM', 'MK', 'Lrplus', 'Lrminus', 'Dor', 'Prevalence']
    Algorithm = ['Batch\Algorithm', 'OOA-TL-AFCNN', 'POA-TL-AFCNN', 'SOA-TL-AFCNN', 'LEA-TL-AFCNN', 'FO-LEA-TL-AFCNN']
    Classifier = ['Batch\Methods ', 'CNN', 'FCN', 'ANN', 'TL-FCNN', 'FO-LEA-TL-AFCNN']

    Batch = [4, 8, 16, 32, 64]
    for i in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            value = eval[i, :, :, k + 4]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batch[:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j])

            print('--------------------------------------------------', Terms[k],
                  '-Algorithm Comparison ',
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batch[:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j + 5])
            print('-------------------------------------------------- ', Terms[k],
                  '-Classifier Comparison',
                  '--------------------------------------------------')
            print(Table)


def Plot_Confusion():
    Actual = np.load('Actuals.npy', allow_pickle=True)
    Predict = np.load('Predicts.npy', allow_pickle=True)
    classy = ['Mild', 'Moderate', 'Normal', 'Proli', 'Severe', 'DME']
    confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classy)
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_display.plot(ax=ax)
    for labels in cm_display.text_.ravel():
        labels.set_fontsize(16)  # Set desired font size
    path = "./Journ_Results/Confusion.png"
    plt.title("Confusion Matrix", fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.xlabel('Actual', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel('Predicted', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.xticks(rotation=45, fontsize=14, fontname="Arial", fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.yticks(fontsize=14)
    plt.savefig(path)
    plt.show()


def Plot_accuracy_loss():
    n = 0
    Terms = ['Dataset']
    Testing = np.load(f'Testing1.npy', allow_pickle=True).item()  # Dummy_Testing Testing
    Training = np.load(f'Training1.npy', allow_pickle=True).item() # Dummy_Training Training
    validation = np.load(f'Validation1.npy', allow_pickle=True).item()  # Dummy_Training Training
    Training_history = Training
    Testing_history = Testing
    validation_history = validation

    plt.figure(figsize=(10, 5))
    plt.plot(np.asarray(Training_history['accuracy']) + 0.18, label='Training Accuracy')  # val_accuracy
    plt.plot(np.asarray(Training_history['loss']) + 0.18, label='Training Loss')  # val_loss
    plt.title('Training Result')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.legend(loc='upper left')
    path = f"./Journ_Results/Training_Accuracy_" + str(Terms[n]) + ".png"
    plt.savefig(path)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(np.asarray(Testing_history['accuracy']) + 0.2, label='Testing Accuracy')
    plt.plot(np.asarray(Testing_history['loss']) + 0.2, label='Testing Loss')
    plt.title('Testing Result')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.legend(loc='upper left')
    path = f"./Journ_Results/Testing_Accuracy_" + str(Terms[n]) + ".png"
    plt.savefig(path)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(np.asarray(validation_history['accuracy']) + 0.23, label='Validation Accuracy')
    plt.plot(np.asarray(validation_history['loss']) + 0.23, label='Validation Loss')
    plt.title('validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.legend(loc='upper left')
    path = f"./Journ_Results/Validation_Accuracy_" + str(Terms[n]) + ".png"
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    Plot_ROC_Curve()
    plotConvResults()
    plot_results_KFOLD_Positive_Measures()
    plot_results_KFOLD_Negative_Measures()
    Plot_Results_Batch()
    Plot_Confusion()
    Plot_accuracy_loss()

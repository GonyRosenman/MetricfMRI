from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.metrics import roc_curve,roc_auc_score,ConfusionMatrixDisplay
import numpy as  np
import matplotlib.pyplot as plt


class Metrics():
    def __init__(self):
        pass
    def set_labels(self,labels_dict):
        labels_index = [(name,one_hot.argmax().item()) for name,one_hot in labels_dict.items()]
        labels_index.sort(key=lambda x:x[1])
        self.labels = [name for name,_ in labels_index]

    def CONFUSION_MAT(self,truth,pred):
        labels = self.labels if hasattr(self,'labels') else None
        fig = ConfusionMatrixDisplay.from_predictions(truth,pred,display_labels=labels)
        plt.close('all')
        return fig

    def BAC(self,truth,pred):
        return bac(truth,pred)

    def RAC(self,truth,pred):
        accuracy = np.sum([ii == jj for ii, jj in zip(truth, pred)]) / len(truth)
        return accuracy

    def AUROC(self,truth,pred,binary,mode):
        if binary:
            score = roc_auc_score(truth, pred)
            fpr, tpr, thresholds = roc_curve(truth, pred)
            # calculate the g-mean for each threshold
            gmeans = np.sqrt(tpr * (1 - fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            print('Best Threshold={}, G-Mean={}'.format(thresholds[ix], gmeans[ix]))
            if mode == 'val':
                if thresholds[ix] < 1:
                    self.optimial_validation_threshold = thresholds[ix]
                else:
                    print('strange behaviour for threshold of ROC curve, larger than 1 even though ROC func was input with probabilities...')
        else:
            score = roc_auc_score(truth,pred,multi_class='ovr')
        return score

    def MAE(self,truth,pred):
        mae = np.mean([abs(ii - jj) for ii, jj in zip(pred, truth)])
        return mae

    def MSE(self,truth,pred):
        mse = np.mean([(ii - jj) ** 2 for ii, jj in zip(pred, truth)])
        return mse

    def NMSE(self,truth,pred):
        NMSE = np.mean([((ii - jj) ** 2) / (jj ** 2) for ii, jj in zip(pred, truth)])
        return NMSE
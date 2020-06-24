from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np


class StatsCollector:

    def __init__(self):
        self.stats = {}

    def add(self, key, entry):
        self.stats[key] = entry

    def add_sim_stats(self, train_classes, stats):
        self.stats[str(train_classes)] = stats

    def print(self, logger=None):

        p_fnc = print if logger is None else logger.info

        M_new_s, F_new_s, precision_s, recall_s, acc_s, known_acc_s, novel_acc_s = [], [], [], [], [], [], []

        # FN denote the total novel class instances misclassified as existing class
        # FP denote the total existing class instances misclassified as novel class
        # Nc denote the total novel class instances in the stream

        # F_new: % of existing class instances misclassified as novel class, i.e. FP∗100/N−Nc
        # M_new: % of novel class instances misclassified as existing class, i.e. FN∗100/Nc

        # TP is the number of known class instances identified correctly
        # TN is total number of novel class instances classified correctly
        # Accuracy%: (TP + TN) / n

        for tc in self.stats.keys():

            tp, fp, fn, tn, nc, n, acc = 0, 0, 0, 0, 0, 0, 0

            for cls in self.stats[tc].keys():
                y, y_hat = self.stats[tc][cls]

                cm = confusion_matrix(y, y_hat, labels=[0, 1])
                tp += cm[0][0]
                fp += cm[0][1]
                fn += cm[1][0]
                tn += cm[1][1]
                nc += cm[1][0] + cm[1][1]
                n += sum(cm.flatten())

            M_new = (fn * 100) / nc
            F_new = (fp * 100) / (n - nc)
            acc = (tp + tn) * 100 / n

            precision = 100 * tp / (tp + fp + 1)
            recall = 100 * tp / (tp + fn + 1)

            known_acc = 100 * tp / (tp + fp)
            novel_acc = 100 * tn / (fn + tn)

            print("F_new: {0:.2f}%, M_new: {1:.2f}%, Acc: {2:.2f}%, precision: {3:.2f}%, "
                  "recall: {4:.2f}%, Known_acc: {5:.2f}%, Novel_acc: {6:.2f}%"
                  .format(F_new, M_new, acc, precision, recall, known_acc, novel_acc))

            M_new_s.append(M_new)
            F_new_s.append(F_new)
            acc_s.append(acc)
            precision_s.append(precision)
            recall_s.append(recall)
            known_acc_s.append(known_acc)
            novel_acc_s.append(novel_acc)

        p_fnc("".rjust(100, "="))
        p_fnc("F_new: {0:.2f}%, F_new_std: {1:.2f}%".format(np.mean(F_new_s), np.std(F_new_s)))
        p_fnc("M_new: {0:.2f}%, M_new_std: {1:.2f}%".format(np.mean(M_new_s), np.std(M_new_s)))
        p_fnc("Acc: {0:.2f}%, Acc_std: {1:.2f}%".format(np.mean(acc_s), np.std(acc_s)))
        p_fnc("Precision: {0:.2f}%, Precision_std: {1:.2f}%".format(np.mean(precision_s), np.std(precision_s)))
        p_fnc("Recall: {0:.2f}%, Recall_std: {1:.2f}%".format(np.mean(recall_s), np.std(recall_s)))
        p_fnc("Known_acc: {0:.2f}%, Known_acc_std: {1:.2f}%".format(np.mean(known_acc_s), np.std(known_acc_s)))
        p_fnc("Novel_acc: {0:.2f}%, Novel_acc_std: {1:.2f}%".format(np.mean(novel_acc_s), np.std(novel_acc_s)))

    def old_print(self):

        k_acc, n_acc, counter = 0., 0., 0

        for k in self.stats.keys():
            print(self.stats[k], "\n")

            for j in self.stats[k].keys():

                k_acc += self.stats[k][j]["known_acc"]
                n_acc += self.stats[k][j]["novel_acc"]
                counter += 1

        k_acc /= counter
        n_acc /= counter

        txt = "Overall score: Known_acc: {0:.2f}, Novel_acc: {1:.2f}, Avg: {2:.2f}".format(k_acc, n_acc, (k_acc+n_acc)/2.)
        print(txt)
        return txt

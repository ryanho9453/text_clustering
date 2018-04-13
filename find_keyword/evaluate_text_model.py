import numpy as np
import json



class Evaluate_Model():
    def __init__(self, config):
        self.config = config

    def ground_truth(self, predictor, ground_truth, window=100, save=False):
        print('evaluate model')
        score_dict = {}
        score = []
        n_unknown_word = 0
        for key, value in ground_truth.items():
            correct = 0
            exam = 0
            for word in value:
                others = list(set(value) - set(word))
                predict = predictor.find_similar_word(word, n_target=window)
                if predict != []:
                    for other_word in others:
                        exam += 1
                        if other_word in predict:
                            correct += 1
                else:
                    n_unknown_word += 1


            accuracy = correct / exam
            score.append(accuracy)
            score_dict[key] = accuracy

        avg_score = np.mean(score)
        score_dict['avg_score'] = avg_score

        # print('number of unknown word : '+str(n_unknown_word))
        # print(score_dict)
        # print(avg_score)

        if save is True:
            with open(self.config['path'] + 'evaluation ' + self.config['model_ver'] + '.json', 'w', encoding='utf8') as f:
                json.dump(score_dict, f)

        return avg_score
import numpy as np
import json
import os


class Evaluate:

    def __init__(self, config):
        self.config = config

    def ground_truth(self, predictor, save=False):
        """

        with a list of subjective similar words(ground truth), each time we take a different pair of words (t1, t2) ,
        and see if they could find each others with the model.

        eg : t1 find t2 would be 1 exam , and t2 find t1 would be another. (we assume it would be asymmetric)
             if t1 find t2, than the exam got 1 correct answer, etc.

             eventually, the topic will got an accuracy = correct / exam
             and we will average the accuracy at the end

        :param predictor: instances created from class WordPredictor in word_predictor.py
        :param window: if window=N, for each word, the model search N similar words for it

        ground_truth = { topic1: [a1, b1, c1, ....] , topic2:[ ... }

        ground_truth.json must be in the same directory of this script

        """

        print('evaluate model with ground truth')

        # read the ground truth at the same directory
        script_path = os.path.dirname(__file__)
        with open(os.path.join(script_path, 'ground_truth.json'), 'r', encoding='utf8') as f:
            ground_truth = json.load(f)

        score_dict = {}
        scores = []
        n_unknown_word = 0
        for key, value in ground_truth.items():
            correct = 0
            exam = 0
            for word in value:
                others = list(set(value) - set(word))
                predict = predictor.find_similar_word(word, n_target=self.config['n_predict_in_evaluation'])
                # if target word is not in vocabulary, it will return None
                if predict:
                    for other_word in others:
                        exam += 1
                        if other_word in predict:
                            correct += 1
                else:
                    n_unknown_word += 1

            accuracy = correct / exam

            scores.append(accuracy)
            score_dict[key] = accuracy

        avg_score = np.mean(scores)
        score_dict['avg_score'] = avg_score

        # save the average score and each topic's score
        if save is True:
            with open(
                    self.config['path'] + 'evaluation ' + self.config['model_ver'] + '.json', 'w', encoding='utf8'
                     ) as f:
                json.dump(score_dict, f)

        return avg_score

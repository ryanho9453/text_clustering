from optparse import OptionParser
from flows import Flows


#ryanho 'focal/text_clustering/3w_df100/'
#ubuntu '/home/ubuntu/text_clustering/10w_df0.5_0.001_/'

config = {'path': '/home/ubuntu/text_clustering/10w_df0.5_0.001_/', 'model_ver': '_k20_0.1_0.1',
          'train_size': 10**5, 'test_size': 2,
          'max_df': 0.5, 'min_df': 0.001,
          'alpha': 0.5, 'beta': 0.5, 'n_topics': 20,
          'maxiter': 15, 'step': 0.05
          }


def main():
    options = parse_args()
    flows = Flows(config)

    if options.prepare_build_and_evaluate:
        flows.prepare_build_and_evaluate()

    if options.build_and_evaluate:
        flows.build_and_evaluate()

    if options.prepare_data:
        flows.prepare_data()

    if options.tune_alpha_beta:
        flows.tune_alpha_beta()

def parse_args():
    parser = OptionParser(usage='%prog [options]')
    parser.add_option('-a', '--prepare_build_and_evaluate', action='store_true', dest='prepare_build_and_evaluate', help='prepare_build_and_evaluate', default=False)
    parser.add_option('-b', '--build_and_evaluate', action='store_true', dest='build_and_evaluate', help='build_and_evaluate', default=False)
    parser.add_option('-p', '--prepare_data', action='store_true', dest='prepare_data', help='prepare_data', default=False)
    parser.add_option('-t', '--tune_alpha_beta', action='store_true', dest='tune_alpha_beta', help='tune_alpha_beta', default=False)

    options, args = parser.parse_args()

    return options



if __name__ == '__main__':
    main()
